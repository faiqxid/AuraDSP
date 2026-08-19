# Music AI Remaster

[English](#english) | [Bahasa Indonesia](#bahasa-indonesia)

---

## English

SOTA AI Music Detection & Remastering Engine designed to detect AI-generated audio (99.88% accuracy) and remaster it to bypass streaming platform detection enforcers (SoundOn, DistroKid, Spotify).

### 🌟 Features

- **SOTA Detection**: Uses `lofcz/ai-music-detector` model (3,585 spectral features) for precise detection.
- **7-Step DSP Remaster Pipeline**:
  1. **Metadata Purge**: Strips ID3/Vorbis/MP4 tags and re-encodes clean audio.
  2. **Stem Separation**: Uses HTDemucs 6s (`audio-separator`) to isolate vocals, drums, bass, guitar, piano, and other stems.
  3. **Humanization**: Applies micro timing offsets and pitch drift with crossfades.
  4. **Per-Stem DSP**: Saturation, chorus, phaser, and peak limiting.
  5. **Stem Recombination**: Gain-staged stereo summation.
  6. **Phase & Spatial Perturbation**: STFT per-bin phase perturbation, gated room tone, and subtle reverb.
  7. **Mastering & Verification**: ITU-R BS.1770-4 LUFS normalization (-14.0 LUFS) + auto-verification before/after AI scores.
- **Reference Track Matching**: Option to match EQ, dynamic range, and target LUFS from a reference audio file.
- **Sequential Batch Folder Mode**: Process entire directories with real-time SSE progress streaming without CUDA OOM errors.
- **Web UI & CLI**: Modern editorial frosted glass UI + CLI support.

### 📊 Detection & Remaster Results

| Track | Mode | Before Remaster | After Remaster | Status |
|-------|------|-----------------|----------------|--------|
| `aigent.mp3` | Quality | 100.0% AI | 0.24% HUMAN | Pass |
| `Total Kinetic Strike 1.mp3` | Aggressive | 100.0% AI | 1.26% HUMAN | Pass |

### 🚀 Quick Start

#### Requirements
- Windows 10/11 or Linux
- Python 3.11+
- NVIDIA GPU (RTX 3050+ recommended, 6GB VRAM) with CUDA support
- FFmpeg installed and in PATH

#### Installation

```bash
git clone https://github.com/your-username/music_ai_remaster.git
cd music_ai_remaster

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ONNX Runtime GPU (DO NOT install CPU-only onnxruntime)
pip install onnxruntime-gpu
```

#### Running the Web Application

```bash
python src/app.py
```
Open `http://127.0.0.1:5000` in your browser.

#### CLI Usage

**Single file AI detection:**
```bash
python src/cli.py detect "path/to/song.mp3"
```

**Single file remaster:**
```bash
python src/cli.py remaster "path/to/song.mp3" --intensity aggressive
```

**Batch folder remaster:**
```bash
python src/cli.py remaster "path/to/folder" --intensity aggressive
```

#### Python API Integration

```python
import sys
sys.path.insert(0, 'src')
from remaster import RemasterPipeline

pipeline = RemasterPipeline(output_dir='output', intensity='aggressive')
result = pipeline.remaster('path/to/ai_song.mp3')

print(f"Output File: {result.filepath}")
print(f"AI Score Before: {result.ai_score_before * 100:.2f}%")
print(f"AI Score After: {result.ai_score_after * 100:.2f}%")
```

---

## Bahasa Indonesia

Engine Deteksi dan Remaster Musik AI SOTA yang dirancang untuk mendeteksi musik buatan AI (akurasi 99.88%) dan meremaster audio tersebut agar lolos dari sistem deteksi platform distribusi musik (SoundOn, DistroKid, Spotify).

### 🌟 Fitur Utama

- **Deteksi SOTA**: Menggunakan model `lofcz/ai-music-detector` (3.585 fitur spektral) untuk deteksi presisi.
- **Pipeline DSP Remaster 7-Tahap**:
  1. **Pembersihan Metadata**: Menghapus tag ID3/Vorbis/MP4 dan melakukan re-encode audio bersih.
  2. **Pemisahan Stem**: Menggunakan HTDemucs 6s (`audio-separator`) untuk memisah stem vokal, drum, bass, gitar, piano, dan instrumen lainnya.
  3. **Humanisasi**: Menerapkan pergeseran timing dan micro pitch drift dengan crossfade.
  4. **DSP Per-Stem**: Saturasi, chorus, phaser, dan peak limiting.
  5. **Penggabungan Kembali Stem**: Penjumlahan audio stereo dengan gain staging yang aman.
  6. **Perturbasi Fase & Spasial**: Perturbasi fase STFT per-bin, gated room tone, dan reverb halus.
  7. **Mastering & Verifikasi**: Normalisasi LUFS standar ITU-R BS.1770-4 (-14.0 LUFS) + otomatis memverifikasi skor AI sebelum & sesudah.
- **Pencocokan Lagu Referensi (Reference Track Matching)**: Opsi untuk menyelaraskan EQ, dynamic range, dan target LUFS sesuai lagu referensi.
- **Modus Batch Folder Sekvensial**: Memproses seluruh folder lagu secara berurutan dengan streaming progress SSE real-time tanpa error CUDA OOM.
- **Tampilan Web & CLI**: Antarmuka web modern dengan gaya editorial frosted glass + dukungan CLI.

### 📊 Hasil Uji Deteksi & Remaster

| Lagu | Mode | Sebelum Remaster | Sesudah Remaster | Status |
|------|------|------------------|------------------|--------|
| `aigent.mp3` | Quality | 100.0% AI | 0.24% HUMAN | Lolos |
| `Total Kinetic Strike 1.mp3` | Aggressive | 100.0% AI | 1.26% HUMAN | Lolos |

### 🚀 Panduan Penggunaan Cepat

#### Persyaratan Sistem
- Windows 10/11 atau Linux
- Python 3.11+
- GPU NVIDIA (Direkomendasikan RTX 3050+, VRAM 6GB) dengan dukungan CUDA
- FFmpeg terinstall dan terdaftar di PATH

#### Cara Instalasi

```bash
git clone https://github.com/username-anda/music_ai_remaster.git
cd music_ai_remaster

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Instalasi PyTorch dengan dukungan CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Instalasi ONNX Runtime GPU (JANGAN install onnxruntime versi CPU)
pip install onnxruntime-gpu
```

#### Menjalankan Aplikasi Web

```bash
python src/app.py
```
Buka `http://127.0.0.1:5000` di browser Anda.

#### Penggunaan via CLI

**Deteksi Musik AI (Satu File):**
```bash
python src/cli.py detect "path/ke/lagu.mp3"
```

**Remaster Audio (Satu File):**
```bash
python src/cli.py remaster "path/ke/lagu.mp3" --intensity aggressive
```

**Remaster Batch Satu Folder:**
```bash
python src/cli.py remaster "path/ke/folder" --intensity aggressive
```

---

## 🛠️ Tech Stack & Architecture

- **Core Audio Processing**: `librosa`, `soundfile`, `scipy`, `numpy`
- **DSP Engine**: `pedalboard` (Spotify DSP Library), `ffmpeg`
- **Stem Separation**: `audio-separator` (HTDemucs 6s model)
- **ML & GPU Acceleration**: PyTorch (`cu121`), `onnxruntime-gpu`, `safetensors`
- **Web Backend**: Flask + Server-Sent Events (SSE)
- **Mastering Standard**: ITU-R BS.1770-4 (`pyloudnorm`) - Target -14.0 LUFS

---

## 📜 License & Disclaimer

Tool ini dibuat untuk tujuan riset dan pengujian kualitas audio. Pengguna bertanggung jawab penuh atas penggunaan materi audio yang diunggah ke platform streaming.
