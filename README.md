# Music AI Remaster

> SOTA AI Music Detection & Remastering Engine — detects AI-generated audio and remasters it to evade streaming platform detection systems.

**[English](#english)** | **[Bahasa Indonesia](#bahasa-indonesia)**

---

<a name="english"></a>
## English

### Overview

Music AI Remaster is a Python tool that combines AI detection and a 7-step DSP remaster pipeline to help AI-generated music bypass streaming platform detection enforcers such as SoundOn, DistroKid, and Spotify.

- **Detection accuracy**: 99.88% (SOTA `lofcz/ai-music-detector` model, 3,585 spectral features)
- **Tested result**: 100% AI → < 2% AI after remaster
- **Hardware**: CUDA-accelerated (NVIDIA GPU recommended)
- **Interface**: Web UI + CLI + Python API

---

### Features

| Feature | Description |
|---------|-------------|
| SOTA Detection | Uses `lofcz/ai-music-detector` logistic regression model on spectral fakeprints |
| 7-Step DSP Pipeline | Metadata purge → stem separation → humanization → spectral DSP → recombination → phase/spatial → mastering |
| Two Intensity Modes | **Quality** (subtle, natural) · **Aggressive** (STFT smoothing + high-frequency vibrato > 2.2 kHz) |
| Reference Track Matching | Match EQ, dynamic range, and LUFS target from a reference audio file (Moises/Ozone-style) |
| LUFS Normalization | ITU-R BS.1770-4 standard, target −14.0 LUFS (SoundOn distribution compliant) |
| Batch Folder Mode | Sequential processing of entire directories with real-time SSE progress streaming |
| No CUDA OOM | Single shared Separator singleton across the full batch sequence |
| Web UI | Editorial frosted glass interface, drag-and-drop, live before/after AI scores |
| CLI | Single file and batch folder remaster with rich progress output |

---

### Remaster Pipeline

```
INPUT: AI-generated .mp3 / .wav
  │
  ├─ 1. Metadata Purge       Strip ID3/Vorbis/MP4 tags, re-encode clean WAV
  ├─ 2. Stem Separation      HTDemucs 6s: vocals / drums / bass / guitar / piano / other
  ├─ 3. Humanization         Micro timing offsets + pitch drift with 64-sample crossfades
  ├─ 4. Spectral DSP         Per-stem: saturation, chorus, phaser, peak limiting
  ├─ 5. Stem Recombination   Gain-staged stereo summation (1/√N scale + 0.85 peak limit)
  ├─ [Aggressive] STFT Smooth + FFmpeg split-band vibrato (> 2.2 kHz, 4 Hz / 0.04 depth)
  ├─ 6. Phase & Spatial      STFT per-bin phase perturbation, gated room tone, subtle reverb
  └─ 7. Mastering & Verify   3-band multiband compression → limiter → −14.0 LUFS → re-detect
```

---

### Test Results

| Track | Intensity | AI Score Before | AI Score After | Result |
|-------|-----------|-----------------|----------------|--------|
| `aigent.mp3` | Quality | 100.0% | 0.24% | Pass |
| `Total Kinetic Strike 1.mp3` | Aggressive | 100.0% | 1.26% | Pass |

---

### Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 or Linux |
| Python | 3.11+ |
| GPU | NVIDIA with CUDA (RTX 3050+ / 6 GB VRAM recommended) |
| FFmpeg | Installed and in system PATH |
| ONNX Runtime | `onnxruntime-gpu` only — **do not** install CPU `onnxruntime` alongside it |

---

### Installation

```bash
git clone https://github.com/your-username/music_ai_remaster.git
cd music_ai_remaster

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime GPU — do NOT install plain onnxruntime
pip install onnxruntime-gpu

# FFmpeg (Windows via winget)
winget install "FFmpeg (Essentials Build)"
```

> **Note**: If you previously installed `onnxruntime` (CPU), uninstall it first:
> `pip uninstall onnxruntime && pip install --force-reinstall onnxruntime-gpu`

---

### Usage

#### Web Application

```bash
python src/app.py
```

Open `http://127.0.0.1:5000`. Supports drag-and-drop single file remaster and batch folder mode with live progress table.

#### CLI

```bash
# Detect a single file
python src/cli.py detect "path/to/song.mp3"

# Remaster a single file
python src/cli.py remaster "path/to/song.mp3" --intensity aggressive

# Batch remaster an entire folder
python src/cli.py remaster "path/to/folder" --intensity aggressive

# Output JSON scores
python src/cli.py detect "path/to/song.mp3" --json-out
```

Remastered outputs are saved to `<folder>/output/<original_name>.wav`.

#### Python API

```python
import sys
sys.path.insert(0, 'src')
from remaster import RemasterPipeline

pipeline = RemasterPipeline(output_dir='output', intensity='aggressive')
result = pipeline.remaster('path/to/ai_song.mp3')

print(f"Output:     {result.filepath}")
print(f"AI Before:  {result.ai_score_before * 100:.2f}%")
print(f"AI After:   {result.ai_score_after * 100:.2f}%")
print(f"Bypassed:   {result.bypass_success}")
```

---

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect` | POST | Analyze uploaded audio file. Field: `file` |
| `/api/remaster` | POST | Remaster single file + auto before/after detect. Fields: `file`, `intensity` |
| `/api/remaster-batch-stream` | POST | SSE streaming batch remaster. Fields: `folder_path`, `intensity` |
| `/api/download/<filename>` | GET | Download remastered WAV |

**SSE Batch Events (`/api/remaster-batch-stream`)**

| Event | Payload |
|-------|---------|
| `scan` | `{ total, files }` |
| `file_start` | `{ index, total, filename }` |
| `step` | `{ index, total, filename, step }` |
| `file_done` | `{ index, total, filename, ai_score_before, ai_score_after, verdict_before, verdict_after, output_path }` |
| `file_failed` | `{ index, total, filename, error }` |
| `done` | `{ total, completed, failed }` |

---

### Project Structure

```
music_ai_remaster/
├── models/                     # SOTA detector weights (safetensors)
├── src/
│   ├── app.py                  # Flask web server + SSE batch endpoint
│   ├── detector.py             # AI detection orchestrator
│   ├── remaster.py             # Remaster pipeline orchestrator
│   ├── cli.py                  # CLI entry point (click + rich)
│   ├── reference_master.py     # Reference track matching (EQ / DR / LUFS)
│   ├── analyzers/              # Detection analyzers (SOTA model, spectral, phase, etc.)
│   ├── processors/             # DSP processors (humanizer, spectral, phase, mastering, etc.)
│   ├── templates/index.html    # Web UI
│   └── utils/                  # Audio I/O utilities
├── tests/                      # Unit tests
├── requirements.txt
└── setup.py
```

---

### Tech Stack

| Layer | Library / Tool |
|-------|---------------|
| Audio I/O | `librosa`, `soundfile` |
| DSP | `pedalboard` (Spotify), `ffmpeg` |
| Stem Separation | `audio-separator` / HTDemucs 6s |
| ML & Inference | `safetensors`, `numpy` |
| GPU Acceleration | `torch 2.5.1+cu121`, `onnxruntime-gpu` |
| Mastering Standard | `pyloudnorm` (ITU-R BS.1770-4) |
| Web Backend | Flask + SSE |
| CLI | `click`, `rich` |
| Metadata | `mutagen` |

---

<a name="bahasa-indonesia"></a>
## Bahasa Indonesia

### Tentang Proyek

Music AI Remaster adalah tool Python yang menggabungkan deteksi AI dengan pipeline DSP 7-tahap untuk meremaster musik buatan AI agar lolos dari sistem deteksi platform distribusi musik seperti SoundOn, DistroKid, dan Spotify.

- **Akurasi deteksi**: 99.88% (model SOTA `lofcz/ai-music-detector`, 3.585 fitur spektral)
- **Hasil uji**: 100% AI → < 2% AI setelah remaster
- **Hardware**: Akselerasi CUDA (GPU NVIDIA direkomendasikan)
- **Antarmuka**: Web UI + CLI + Python API

---

### Fitur Utama

| Fitur | Deskripsi |
|-------|-----------|
| Deteksi SOTA | Menggunakan model regresi logistik `lofcz/ai-music-detector` pada spectral fakeprint |
| Pipeline DSP 7-Tahap | Purge metadata → pemisahan stem → humanisasi → DSP spektral → rekombinasi → fase/spasial → mastering |
| Dua Mode Intensitas | **Quality** (halus, natural) · **Aggressive** (STFT smoothing + vibrato frekuensi tinggi > 2,2 kHz) |
| Pencocokan Lagu Referensi | Sesuaikan EQ, dynamic range, dan target LUFS dari file audio referensi (gaya Moises/Ozone) |
| Normalisasi LUFS | Standar ITU-R BS.1770-4, target −14,0 LUFS (sesuai standar distribusi SoundOn) |
| Mode Batch Folder | Pemrosesan sekuensial seluruh folder dengan streaming progress SSE real-time |
| Tanpa CUDA OOM | Satu instance Separator bersama di seluruh sesi batch |
| Web UI | Tampilan editorial frosted glass, drag-and-drop, skor AI before/after langsung |
| CLI | Remaster satu file dan batch folder dengan tampilan progress kaya (rich) |

---

### Persyaratan Sistem

| Komponen | Kebutuhan |
|----------|-----------|
| Sistem Operasi | Windows 10/11 atau Linux |
| Python | 3.11+ |
| GPU | NVIDIA dengan CUDA (RTX 3050+ / VRAM 6 GB direkomendasikan) |
| FFmpeg | Terinstall dan terdaftar di PATH sistem |
| ONNX Runtime | `onnxruntime-gpu` saja — **jangan** install `onnxruntime` CPU bersamaan |

---

### Instalasi

```bash
git clone https://github.com/username-anda/music_ai_remaster.git
cd music_ai_remaster

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# PyTorch dengan CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# ONNX Runtime GPU — JANGAN install plain onnxruntime
pip install onnxruntime-gpu

# FFmpeg (Windows via winget)
winget install "FFmpeg (Essentials Build)"
```

> **Catatan**: Jika sebelumnya sudah menginstall `onnxruntime` (CPU), uninstall terlebih dahulu:
> `pip uninstall onnxruntime && pip install --force-reinstall onnxruntime-gpu`

---

### Penggunaan

#### Aplikasi Web

```bash
python src/app.py
```

Buka `http://127.0.0.1:5000`. Mendukung drag-and-drop remaster satu file dan mode batch folder dengan tabel progress live.

#### CLI

```bash
# Deteksi satu file
python src/cli.py detect "path/ke/lagu.mp3"

# Remaster satu file
python src/cli.py remaster "path/ke/lagu.mp3" --intensity aggressive

# Remaster batch satu folder
python src/cli.py remaster "path/ke/folder" --intensity aggressive

# Output skor dalam format JSON
python src/cli.py detect "path/ke/lagu.mp3" --json-out
```

Hasil remaster disimpan di `<folder>/output/<nama_asli>.wav`.

#### Python API

```python
import sys
sys.path.insert(0, 'src')
from remaster import RemasterPipeline

pipeline = RemasterPipeline(output_dir='output', intensity='aggressive')
result = pipeline.remaster('path/to/ai_song.mp3')

print(f"File Output:  {result.filepath}")
print(f"AI Sebelum:   {result.ai_score_before * 100:.2f}%")
print(f"AI Sesudah:   {result.ai_score_after * 100:.2f}%")
print(f"Bypass Lolos: {result.bypass_success}")
```

---

### Hasil Uji

| Lagu | Mode | Skor AI Sebelum | Skor AI Sesudah | Hasil |
|------|------|-----------------|-----------------|-------|
| `aigent.mp3` | Quality | 100,0% | 0,24% | Lolos |
| `Total Kinetic Strike 1.mp3` | Aggressive | 100,0% | 1,26% | Lolos |

---

## Disclaimer / Penafian

This tool is built for audio research and quality testing purposes. Users are fully responsible for how remastered audio is distributed across streaming platforms.

Tool ini dibuat untuk tujuan riset dan pengujian kualitas audio. Pengguna bertanggung jawab penuh atas distribusi audio hasil remaster ke platform streaming.
