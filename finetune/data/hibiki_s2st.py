import json
import math
from pathlib import Path

import numpy as np
import sphn
import torch
from moshi.conditioners import ConditionAttributes

from .interleaver import Alignment, Sample


VALID_SPEAKER_SIMILARITY_LABELS = {
    "very_bad",
    "bad",
    "neutral",
    "good",
    "very_good",
}

REQUIRED_FIELDS = {
    "source_audio",
    "target_audio",
    "duration",
    "target_alignment",
}


def normalize_speaker_similarity(label: object) -> str:
    if isinstance(label, str):
        normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in VALID_SPEAKER_SIMILARITY_LABELS:
            return normalized
    return "neutral"


def make_condition_attributes(label: object) -> ConditionAttributes:
    return ConditionAttributes(
        text={"description": normalize_speaker_similarity(label)},
        tensor={},
    )


def resolve_manifest_path(path: str, manifest_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return manifest_dir / candidate


def _validate_row(row: dict) -> None:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"Hibiki S2ST row is missing required fields: {missing}")


def _as_mono(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 1:
        return wav[None, :]
    if wav.ndim == 2:
        return wav.mean(axis=0, keepdims=True)
    raise ValueError(f"Expected a 1D or 2D waveform, got shape {wav.shape}")


def _slice_alignments(
    alignments: list[Alignment], start_sec: float, duration_sec: float
) -> list[Alignment]:
    end_sec = start_sec + duration_sec
    sliced = []
    for word, timestamps, speaker in alignments:
        word_start, word_end = timestamps
        if word_start >= end_sec or word_end <= start_sec:
            continue
        sliced.append(
            (
                word,
                (max(0.0, word_start - start_sec), max(0.0, word_end - start_sec)),
                speaker,
            )
        )
    return sliced


class HibikiS2STTokenizer:
    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        n_q: int,
        dep_q: int,
        expected_num_codebooks: int,
    ):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.n_q = n_q
        self.dep_q = dep_q
        self.expected_num_codebooks = expected_num_codebooks
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)

    def __call__(self, row: dict, start_sec: float, manifest_dir: Path) -> Sample:
        _validate_row(row)
        row_duration = float(row["duration"])
        segment_duration_sec = min(
            self.duration_sec, max(0.0, row_duration - start_sec)
        )
        if segment_duration_sec <= 0:
            raise ValueError(
                f"Invalid segment duration {segment_duration_sec} at {start_sec}"
            )

        source_audio = resolve_manifest_path(row["source_audio"], manifest_dir)
        target_audio = resolve_manifest_path(row["target_audio"], manifest_dir)
        alignment_path = resolve_manifest_path(row["target_alignment"], manifest_dir)

        with torch.no_grad():
            source_audio_tokens = self._encode_audio(source_audio, start_sec)
            target_audio_tokens = self._encode_audio(target_audio, start_sec)

            with alignment_path.open() as f:
                alignment_data = json.load(f)
            alignments = _slice_alignments(
                alignment_data["alignments"], start_sec, self.duration_sec
            )
            target_text = self.interleaver.prepare_item(
                alignments, segment_duration_sec
            )
            target_text = self._pad_or_truncate(
                target_text, value=self.interleaver.zero_padding
            )

            target_audio = target_audio_tokens[:, : self.dep_q, :]
            source_audio = source_audio_tokens[:, : self.n_q - self.dep_q, :]
            codes = torch.cat([target_text, target_audio, source_audio], dim=1)

        if codes.shape[1] != self.expected_num_codebooks:
            raise ValueError(
                f"Expected {self.expected_num_codebooks} streams, got {codes.shape[1]}"
            )
        return Sample(codes, make_condition_attributes(row.get("speaker_similarity")))

    def _encode_audio(self, path: Path, start_sec: float) -> torch.Tensor:
        wav, _ = sphn.read(
            path,
            start_sec=start_sec,
            duration_sec=self.duration_sec,
            sample_rate=self.mimi.sample_rate,
        )
        wav = _as_mono(wav)
        device = getattr(self.interleaver, "device", "cuda")
        audio_tensor = torch.as_tensor(wav, dtype=torch.float32, device=device)
        audio_tokens = self.mimi.encode(audio_tensor[:, None])
        audio_tokens = self._pad_or_truncate(
            audio_tokens, value=self.interleaver.zero_padding
        )
        return audio_tokens.view(1, -1, self.num_audio_frames)

    def _pad_or_truncate(self, tensor: torch.Tensor, value: int) -> torch.Tensor:
        tensor = tensor[..., : self.num_audio_frames]
        num_frames = tensor.shape[-1]
        if num_frames == self.num_audio_frames:
            return tensor
        return torch.nn.functional.pad(
            tensor,
            (0, self.num_audio_frames - num_frames),
            value=value,
        )
