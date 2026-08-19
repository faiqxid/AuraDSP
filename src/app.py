import os
import sys
import tempfile
import warnings
import dataclasses
import json
import queue
import threading
import traceback
import shutil
import time
from pathlib import Path
from collections import deque
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context

# --- CLI Log Capture System ---
class LogCapture:
    def __init__(self):
        self.listeners = []
        self.history = deque(maxlen=500)
        self.lock = threading.Lock()
        
    def write(self, text):
        if not text:
            return
        with self.lock:
            self.history.append(text)
            for q in self.listeners:
                try:
                    q.put_nowait(text)
                except queue.Full:
                    pass
                    
    def flush(self):
        pass

log_capture = LogCapture()

class TeeStream:
    def __init__(self, original_stream, capture):
        self.original_stream = original_stream
        self.capture = capture
        
    def write(self, text):
        self.original_stream.write(text)
        self.capture.write(text)
        
    def flush(self):
        self.original_stream.flush()
        self.capture.flush()

sys.stdout = TeeStream(sys.stdout, log_capture)
sys.stderr = TeeStream(sys.stderr, log_capture)

# Silence python level warnings before importing detector
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from detector import AIMusicDetector
from utils.frozen import get_base_dir

base_dir = get_base_dir()
template_dir = base_dir / "src" / "templates"
static_dir = base_dir / "src" / "static"

app = Flask(__name__, template_folder=str(template_dir), static_folder=str(static_dir))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max upload

# Ensure upload directory exists
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "auradsp_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

shutdown_event = threading.Event()
_REM_OUTPUTS_DIR = UPLOAD_FOLDER / "remaster_outputs"
_SWEEPER_STARTED = False
_OUTPUT_MAX_AGE_SECONDS = 60 * 60
_SWEEPER_INTERVAL_SECONDS = 10 * 60


def cleanup_stale_remaster_outputs(max_age_seconds: int = _OUTPUT_MAX_AGE_SECONDS) -> int:
    """Delete stale single-track remaster WAV downloads from temp storage."""
    if not _REM_OUTPUTS_DIR.exists():
        return 0
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for wav_path in _REM_OUTPUTS_DIR.glob("*.wav"):
        try:
            if wav_path.is_file() and wav_path.stat().st_mtime < cutoff:
                wav_path.unlink()
                deleted += 1
        except Exception:
            pass
    return deleted


def start_cleanup_sweeper() -> None:
    """Start one daemon sweeper that stops when shutdown_event is set."""
    global _SWEEPER_STARTED
    if _SWEEPER_STARTED:
        return
    _SWEEPER_STARTED = True

    def _run():
        cleanup_stale_remaster_outputs()
        while not shutdown_event.wait(_SWEEPER_INTERVAL_SECONDS):
            cleanup_stale_remaster_outputs()

    threading.Thread(target=_run, daemon=True, name="remaster-output-sweeper").start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"}), 400

    # Save to temp path
    temp_path = UPLOAD_FOLDER / f"upload_{os.urandom(8).hex()}{suffix}"
    try:
        file.save(str(temp_path))

        detector = AIMusicDetector(sample_rate=22050, max_seconds=180.0)
        res = detector.detect(str(temp_path))

        return jsonify(dataclasses.asdict(res))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.route("/api/remaster", methods=["POST"])
def remaster():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_extensions:
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(allowed_extensions)}"}), 400

    upload_path = UPLOAD_FOLDER / f"upload_{os.urandom(8).hex()}{suffix}"
    ref_path: Path | None = None
    try:
        file.save(str(upload_path))

        from remaster import RemasterPipeline
        from detector import AIMusicDetector

        intensity = request.form.get("intensity", "aggressive")
        ref_file = request.files.get("reference", None)
        if ref_file:
            ref_suffix = Path(ref_file.filename).suffix.lower()
            if ref_suffix not in {".wav", ".flac", ".mp3"}:
                ref_suffix = ".wav"
            ref_path = UPLOAD_FOLDER / f"ref_{os.urandom(8).hex()}{ref_suffix}"
            ref_file.save(str(ref_path))

        # Detect original BEFORE remaster
        detector = AIMusicDetector(sample_rate=22050, max_seconds=180.0)
        before_res = detector.detect(str(upload_path))

        pipeline = RemasterPipeline(
            output_dir=UPLOAD_FOLDER / "remaster_outputs",
            sample_rate=44100,
            use_cuda=True,
            intensity=intensity,
            shutdown_event=shutdown_event,
        )

        result = pipeline.remaster(str(upload_path), reference_path=ref_path)

        # Auto-analyze remastered output
        after_res = detector.detect(result.filepath)

        response = dataclasses.asdict(result)
        response["ai_score_before"] = before_res.percentage
        response["ai_score_after"] = after_res.percentage
        response["verdict_before"] = before_res.verdict
        response["verdict_after"] = after_res.verdict
        response["vector_scores_before"] = before_res.vector_scores
        response["vector_scores_after"] = after_res.vector_scores
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if upload_path.exists():
            try:
                upload_path.unlink()
            except Exception:
                pass
        if ref_path and ref_path.exists():
            try:
                ref_path.unlink()
            except Exception:
                pass


@app.route("/api/download/<filename>", methods=["GET"])
def download_remaster(filename):
    """Serve remastered output files."""
    # Prevent directory traversal
    safe_name = Path(filename).name
    file_path = UPLOAD_FOLDER / "remaster_outputs" / safe_name
    if not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=safe_name)


@app.route("/api/remaster-batch", methods=["POST"])
def remaster_batch():
    """Batch remaster all audio files in a folder.

    Form data:
        folder_path: absolute path to folder containing audio files
        intensity: 'quality' or 'aggressive' (default: aggressive)

    Returns JSON with total, completed, failed, and per-file results.
    """
    folder_path_str = request.form.get("folder_path", "").strip()
    if not folder_path_str:
        return jsonify({"error": "folder_path is required"}), 400

    folder_path = Path(folder_path_str)
    if not folder_path.is_dir():
        return jsonify({"error": f"Not a valid folder: {folder_path}"}), 400

    from remaster import RemasterPipeline
    from detector import AIMusicDetector

    intensity = request.form.get("intensity", "aggressive")
    ref_file = request.files.get("reference", None)
    batch_ref_path: Path | None = None
    if ref_file:
        ref_suffix = Path(ref_file.filename).suffix.lower()
        if ref_suffix not in {".wav", ".flac", ".mp3"}:
            ref_suffix = ".wav"
        batch_ref_path = UPLOAD_FOLDER / f"ref_{os.urandom(8).hex()}{ref_suffix}"
        ref_file.save(str(batch_ref_path))

    pipeline = RemasterPipeline(
        output_dir=folder_path / "output",
        sample_rate=44100,
        use_cuda=True,
        intensity=intensity,
        shutdown_event=shutdown_event,
    )

    detector = AIMusicDetector(sample_rate=22050, max_seconds=180.0)

    try:
        results = pipeline.batch_remaster(str(folder_path), reference_path=batch_ref_path)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if batch_ref_path and batch_ref_path.exists():
            try:
                batch_ref_path.unlink()
            except Exception:
                pass

    completed = 0
    failed = 0
    response_results = []

    for result in results:
        if result.filepath:
            completed += 1
            try:
                before_res = detector.detect(result.original_path)
                after_res = detector.detect(result.filepath)
                response_results.append({
                    "filename": Path(result.original_path).name,
                    "input_path": result.original_path,
                    "output_path": result.filepath,
                    "status": "pass",
                    "ai_score_before": before_res.percentage,
                    "ai_score_after": after_res.percentage,
                    "verdict_before": before_res.verdict,
                    "verdict_after": after_res.verdict,
                })
            except Exception as e:
                response_results.append({
                    "filename": Path(result.original_path).name,
                    "input_path": result.original_path,
                    "output_path": result.filepath,
                    "status": "pass",  # remastered but post-analysis failed
                    "ai_score_before": None,
                    "ai_score_after": None,
                    "verdict_before": None,
                    "verdict_after": None,
                    "error": str(e),
                })
        else:
            failed += 1
            response_results.append({
                "filename": Path(result.original_path).name,
                "input_path": result.original_path,
                "output_path": "",
                "status": "fail",
                "ai_score_before": None,
                "ai_score_after": None,
                "verdict_before": None,
                "verdict_after": None,
                "error": "; ".join(result.notes) if result.notes else "Unknown error",
            })

    return jsonify({
        "total": len(results),
        "completed": completed,
        "failed": failed,
        "folder_path": str(folder_path),
        "intensity": intensity,
        "results": response_results,
    })


@app.route("/api/remaster-batch-stream", methods=["POST"])
def remaster_batch_stream():
    """Server-Sent Events stream for sequential batch remaster progress.

    Form data:
        folder_path: absolute path to folder containing audio files
        intensity: 'quality' or 'aggressive' (default: aggressive)

    Streams SSE events:
        - scan:       {total: N, files: [...]}
        - file_start: {index, total, filename}
        - step:       {index, total, filename, step}
        - file_done:  {index, total, filename, ai_score_before, ai_score_after}
        - file_failed:{index, total, filename, error}
        - done:       {total, completed, failed}
    """
    folder_path_str = request.form.get("folder_path", "").strip()
    if not folder_path_str:
        return jsonify({"error": "folder_path is required"}), 400

    folder_path = Path(folder_path_str)
    if not folder_path.is_dir():
        return jsonify({"error": f"Not a valid folder: {folder_path}"}), 400

    intensity = request.form.get("intensity", "aggressive")
    ref_file = request.files.get("reference", None)
    stream_ref_path: Path | None = None
    if ref_file:
        ref_suffix = Path(ref_file.filename).suffix.lower()
        if ref_suffix not in {".wav", ".flac", ".mp3"}:
            ref_suffix = ".wav"
        stream_ref_path = UPLOAD_FOLDER / f"ref_{os.urandom(8).hex()}{ref_suffix}"
        ref_file.save(str(stream_ref_path))

    audio_extensions = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    )

    if not audio_files:
        return jsonify({"error": f"No audio files found in: {folder_path}"}), 400

    event_queue: queue.Queue = queue.Queue()
    SENTINEL = object()

    def emit(event_data: dict):
        event_queue.put(event_data)

    def worker():
        try:
            from remaster import RemasterPipeline
            from detector import AIMusicDetector

            emit({"type": "scan", "total": len(audio_files),
                  "files": [f.name for f in audio_files]})

            final_output = folder_path / "output"
            final_output.mkdir(exist_ok=True)
            detector = AIMusicDetector(sample_rate=22050, max_seconds=180.0)
            completed = 0
            failed = 0

            pipeline = RemasterPipeline(
                output_dir=final_output,
                sample_rate=44100,
                use_cuda=True,
                intensity=intensity,
                shutdown_event=shutdown_event,
            )

            for i, audio_file in enumerate(audio_files, 1):
                emit({"type": "file_start", "index": i, "total": len(audio_files),
                      "filename": audio_file.name})

                try:
                    def progress_cb(event, _i=i, _af=audio_file):
                        emit({**event, "index": _i, "total": len(audio_files),
                              "filename": _af.name})

                    # Detect before remaster
                    before_res = detector.detect(str(audio_file))

                    # Remaster (sequential, no locks needed)
                    result = pipeline.remaster(
                        str(audio_file),
                        output_name=f"{audio_file.stem}.wav",
                        progress_callback=progress_cb,
                        reference_path=stream_ref_path,
                    )

                    # Detect after remaster
                    after_res = detector.detect(result.filepath)

                    completed += 1
                    emit({
                        "type": "file_done",
                        "index": i,
                        "total": len(audio_files),
                        "filename": audio_file.name,
                        "output_path": result.filepath,
                        "ai_score_before": before_res.percentage,
                        "ai_score_after": after_res.percentage,
                        "verdict_before": before_res.verdict,
                        "verdict_after": after_res.verdict,
                    })
                except Exception as e:
                    failed += 1
                    traceback.print_exc()
                    emit({
                        "type": "file_failed",
                        "index": i,
                        "total": len(audio_files),
                        "filename": audio_file.name,
                        "error": str(e),
                    })
                    from remaster import CancellationError  # avoid circular import at module level
                    if isinstance(e, CancellationError):
                        break  # abort remaining files, let outer finally handle cleanup

            emit({"type": "done", "total": len(audio_files),
                  "completed": completed, "failed": failed})
        except Exception as e:
            traceback.print_exc()
            emit({"type": "error", "error": str(e)})
        finally:
            event_queue.put(SENTINEL)
            if stream_ref_path and stream_ref_path.exists():
                try:
                    stream_ref_path.unlink()
                except Exception:
                    pass

    def generate():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        while True:
            item = event_queue.get()
            if item is SENTINEL:
                break
            yield f"data: {json.dumps(item)}\n\n"

        thread.join(timeout=1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/api/logs")
def stream_logs():
    """Server-Sent Events endpoint for raw CLI logs."""
    def generate():
        q = queue.Queue(maxsize=100)
        with log_capture.lock:
            log_capture.listeners.append(q)
            history = list(log_capture.history)
        for line in history:
            yield f"data: {json.dumps(line)}\n\n"
        try:
            while True:
                # Ping every second to keep connection alive
                try:
                    line = q.get(timeout=1.0)
                    yield f"data: {json.dumps(line)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            with log_capture.lock:
                if q in log_capture.listeners:
                    log_capture.listeners.remove(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )

if __name__ == "__main__":
    start_cleanup_sweeper()
    app.run(host="127.0.0.1", port=5000, debug=True)
