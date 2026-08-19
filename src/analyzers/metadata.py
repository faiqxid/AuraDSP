"""Metadata Analyzer — Detect AI-related metadata tags and watermarks.

AI music generators embed various metadata markers:
- C2PA Content Credentials (cryptographically signed provenance)
- SynthID watermarks (Google, embedded in waveform)
- Generator-specific tags in ID3/Vorbis/MP4 headers
- "Made with AI" labels in EXIF/XMP
- Suspicious encoding patterns

This analyzer checks file metadata without modifying it.
"""
import os
from pathlib import Path
from dataclasses import dataclass

try:
    import mutagen
    from mutagen.id3 import ID3
    from mutagen.flac import FLAC
    from mutagen.mp4 import MP4
    from mutagen.oggvorbis import OggVorbis
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False


@dataclass
class MetadataResult:
    score: float  # 0.0 = definitely human, 1.0 = definitely AI
    confidence: str  # "low", "medium", "high"
    details: dict
    verdict: str  # "human", "ai", "uncertain"


class MetadataAnalyzer:
    """Analyze file metadata for AI generation markers."""

    # Known AI-related keywords in metadata
    AI_KEYWORDS = [
        "suno", "udio", "suno.ai", "udio.ai", "musicgen", "musiclm",
        "stable audio", "riffusion", "bark", "mubert", "aiva", "amper",
        "boomy", "soundraw", "beatoven", "muse", "lalal.ai",
        "artificial", "synthetic", "generated", "ai-generated",
        "machine learning", "neural", "deep learning", "gpt",
        "made with ai", "created with ai", "produced by ai",
        "synthid", "c2pa", "content credentials",
    ]

    # Known AI generator file signatures / encoding patterns
    SUSPICIOUS_ENCODERS = [
        "lame",  # Some AI tools use specific lame versions
    ]

    def analyze(self, filepath: str) -> MetadataResult:
        """Analyze metadata for AI-related markers."""
        p = Path(filepath)
        if not p.exists():
            return MetadataResult(
                score=0.0, confidence="low", verdict="uncertain",
                details={"error": f"File not found: {filepath}"},
            )

        findings = []
        ai_signals = 0
        total_checks = 0

        # --- Check 1: File-level metadata via mutagen ---
        if HAS_MUTAGEN:
            mutagen_result = self._check_mutagen(str(p))
            ai_signals += mutagen_result["ai_signals"]
            total_checks += mutagen_result["total_checks"]
            findings.extend(mutagen_result["findings"])

        # --- Check 2: Raw binary scan for known AI markers ---
        binary_result = self._check_binary(str(p))
        ai_signals += binary_result["ai_signals"]
        total_checks += binary_result["total_checks"]
        findings.extend(binary_result["findings"])

        # --- Check 3: C2PA / Content Credentials ---
        c2pa_result = self._check_c2pa(str(p))
        ai_signals += c2pa_result["ai_signals"]
        total_checks += c2pa_result["total_checks"]
        findings.extend(c2pa_result["findings"])

        # --- Check 4: Filename patterns ---
        fname = p.name.lower()
        for kw in self.AI_KEYWORDS:
            if kw in fname:
                findings.append(f"Filename contains AI keyword: '{kw}'")
                ai_signals += 1
                total_checks += 1
                break

        # --- Score ---
        if total_checks == 0:
            return MetadataResult(
                score=0.1, confidence="low", verdict="uncertain",
                details={"findings": ["No metadata to analyze"], "total_checks": 0},
            )

        # Metadata-only score is more conservative (metadata can be stripped)
        raw_score = ai_signals / total_checks
        # Cap at 0.8 — metadata alone isn't definitive (could be stripped from human audio)
        final_score = min(0.8, raw_score)

        # But if we find strong AI markers (Suno tags, C2PA AI), bump up
        strong_markers = [f for f in findings if any(
            kw in f.lower() for kw in ["suno", "udio", "c2pa", "synthid", "ai-generated"]
        )]
        if strong_markers:
            final_score = min(1.0, final_score + 0.2)

        # Verdict — metadata is supportive evidence, not definitive
        if final_score >= 0.50:
            verdict = "ai"
            confidence = "high" if final_score >= 0.70 else "medium"
        elif final_score <= 0.15:
            verdict = "human"
            confidence = "low"  # Absence of AI metadata doesn't prove human
        else:
            verdict = "uncertain"
            confidence = "low"

        return MetadataResult(
            score=round(final_score, 4),
            confidence=confidence,
            verdict=verdict,
            details={
                "findings": findings,
                "ai_signals": ai_signals,
                "total_checks": total_checks,
                "raw_score": round(raw_score, 4),
                "strong_markers": len(strong_markers),
            },
        )

    def _check_mutagen(self, filepath: str) -> dict:
        """Check metadata using mutagen library."""
        ai_signals = 0
        total_checks = 0
        findings = []

        try:
            audio = mutagen.File(filepath)
            if audio is None:
                return {"ai_signals": 0, "total_checks": 0, "findings": []}

            # Collect all tag values
            all_tags = {}
            if hasattr(audio, "tags") and audio.tags:
                for key in audio.tags:
                    try:
                        val = str(audio.tags[key])
                        all_tags[str(key)] = val
                    except Exception:
                        pass

            # Check each tag value against AI keywords
            for tag_key, tag_val in all_tags.items():
                val_lower = tag_val.lower()
                for kw in self.AI_KEYWORDS:
                    if kw in val_lower:
                        findings.append(f"Tag '{tag_key}' contains AI keyword: '{kw}' (value: {tag_val[:80]})")
                        ai_signals += 1
                        break
                total_checks += 1

            # Check encoder info
            if hasattr(audio, "info"):
                info = audio.info
                if hasattr(info, "encoder"):
                    enc = str(info.encoder).lower()
                    for kw in self.AI_KEYWORDS:
                        if kw in enc:
                            findings.append(f"Encoder contains AI keyword: '{enc}'")
                            ai_signals += 1
                            break
                    total_checks += 1

        except Exception as e:
            findings.append(f"Mutagen parse error: {str(e)[:60]}")

        return {"ai_signals": ai_signals, "total_checks": total_checks, "findings": findings}

    def _check_binary(self, filepath: str) -> dict:
        """Scan first/last 8KB of file for AI-related strings."""
        ai_signals = 0
        total_checks = 0
        findings = []

        try:
            with open(filepath, "rb") as f:
                # Read head + tail
                head = f.read(8192)
                if os.path.getsize(filepath) > 16384:
                    f.seek(-8192, 2)
                tail = f.read(8192)
                data = head + tail

            # Decode as latin-1 (preserves all bytes) and search
            text = data.decode("latin-1", errors="ignore").lower()

            for kw in self.AI_KEYWORDS:
                if kw in text:
                    findings.append(f"Binary scan found AI keyword: '{kw}'")
                    ai_signals += 1
                    total_checks += 1

            # Check for C2PA magic bytes
            if b"C2PA" in data or b"c2pa" in data:
                findings.append("C2PA content credential marker found in binary")
                ai_signals += 1
            total_checks += 1

        except Exception as e:
            findings.append(f"Binary scan error: {str(e)[:60]}")

        return {"ai_signals": ai_signals, "total_checks": total_checks, "findings": findings}

    def _check_c2pa(self, filepath: str) -> dict:
        """Check for C2PA Content Credentials."""
        ai_signals = 0
        total_checks = 1
        findings = []

        try:
            with open(filepath, "rb") as f:
                data = f.read(min(65536, os.path.getsize(filepath)))

            # C2PA markers
            c2pa_markers = [b"C2PA", b"c2pa", b"jumbf", b"JUMBF", b"contentcredentials"]
            for marker in c2pa_markers:
                if marker in data:
                    findings.append(f"C2PA marker found: {marker.decode('latin-1')}")
                    ai_signals += 1
                    break

            # Look for JSON-LD with AI provenance
            try:
                text = data.decode("utf-8", errors="ignore")
                if '"@type"' in text and ("ai" in text.lower() or "generated" in text.lower()):
                    findings.append("JSON-LD with AI-related type found")
                    ai_signals += 1
            except Exception:
                pass

        except Exception as e:
            findings.append(f"C2PA check error: {str(e)[:60]}")

        return {"ai_signals": ai_signals, "total_checks": total_checks, "findings": findings}
