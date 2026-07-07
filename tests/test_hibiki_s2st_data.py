import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch


@dataclass
class ConditionAttributes:
    text: dict = field(default_factory=dict)
    tensor: dict = field(default_factory=dict)


moshi_module = types.ModuleType("moshi")
conditioners_module = types.ModuleType("moshi.conditioners")
conditioners_module.ConditionAttributes = ConditionAttributes
sys.modules["moshi"] = moshi_module
sys.modules["moshi.conditioners"] = conditioners_module

from finetune.data.args import DataArgs
from finetune.data.hibiki_s2st import (
    HibikiS2STTokenizer,
    normalize_speaker_similarity,
)


class FakeMimi:
    sample_rate = 24000
    frame_rate = 12.5

    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def encode(self, audio):
        batch = audio.shape[0]
        frames = max(1, math.ceil(audio.shape[-1] / 1920))
        values = torch.arange(
            self.num_codebooks * frames,
            device=audio.device,
            dtype=torch.long,
        )
        return values.view(1, self.num_codebooks, frames).repeat(batch, 1, 1)


class FakeInterleaver:
    device = "cpu"
    zero_padding = -1

    def __init__(self):
        self.last_segment_duration = None

    def prepare_item(self, alignments, segment_duration):
        self.last_segment_duration = segment_duration
        frames = math.ceil(segment_duration * FakeMimi.frame_rate)
        return torch.full((1, 1, frames), 101, dtype=torch.long)


def write_manifest_files(tmp_path: Path, duration: float = 1.0) -> dict:
    alignment_path = tmp_path / "target_alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "alignments": [
                    ["hello", [0.10, 0.30], "SPEAKER_MAIN"],
                    ["world", [0.35, 0.55], "SPEAKER_MAIN"],
                ]
            }
        )
    )
    return {
        "source_audio": "source.wav",
        "target_audio": "target.wav",
        "duration": duration,
        "target_alignment": "target_alignment.json",
        "speaker_similarity": "good",
    }


def patch_audio_read(monkeypatch):
    def fake_read(*args, **kwargs):
        return np.zeros((2, 24000), dtype=np.float32), 24000

    monkeypatch.setattr("finetune.data.hibiki_s2st.sphn.read", fake_read)


def build_tokenizer(n_q: int, dep_q: int) -> tuple[HibikiS2STTokenizer, FakeInterleaver]:
    interleaver = FakeInterleaver()
    tokenizer = HibikiS2STTokenizer(
        mimi=FakeMimi(num_codebooks=dep_q),
        interleaver=interleaver,
        duration_sec=1.0,
        n_q=n_q,
        dep_q=dep_q,
        expected_num_codebooks=1 + n_q,
    )
    return tokenizer, interleaver


def test_normalize_speaker_similarity_accepts_valid_labels():
    assert normalize_speaker_similarity("very_bad") == "very_bad"
    assert normalize_speaker_similarity("very bad") == "very_bad"
    assert normalize_speaker_similarity("VERY-GOOD") == "very_good"
    assert normalize_speaker_similarity(" good ") == "good"


def test_normalize_speaker_similarity_maps_missing_and_unknown_to_neutral():
    assert normalize_speaker_similarity(None) == "neutral"
    assert normalize_speaker_similarity("excellent") == "neutral"
    assert normalize_speaker_similarity(3) == "neutral"


def test_data_args_default_mode_is_moshi():
    assert DataArgs().mode == "moshi"


def test_data_args_rejects_unknown_mode():
    with pytest.raises(ValueError, match="data.mode"):
        DataArgs(mode="unknown")


def test_hibiki_tokenizer_emits_17_streams_for_1b_shape(tmp_path, monkeypatch):
    patch_audio_read(monkeypatch)
    row = write_manifest_files(tmp_path)
    tokenizer, _ = build_tokenizer(n_q=16, dep_q=8)

    sample = tokenizer(row, 0.0, tmp_path)

    assert list(sample.codes.shape) == [1, 17, 13]


def test_hibiki_tokenizer_emits_33_streams_for_2b_shape(tmp_path, monkeypatch):
    patch_audio_read(monkeypatch)
    row = write_manifest_files(tmp_path)
    tokenizer, _ = build_tokenizer(n_q=32, dep_q=16)

    sample = tokenizer(row, 0.0, tmp_path)

    assert list(sample.codes.shape) == [1, 33, 13]


def test_hibiki_tokenizer_returns_description_condition_attributes(
    tmp_path, monkeypatch
):
    patch_audio_read(monkeypatch)
    row = write_manifest_files(tmp_path)
    row["speaker_similarity"] = "very-good"
    tokenizer, _ = build_tokenizer(n_q=16, dep_q=8)

    sample = tokenizer(row, 0.0, tmp_path)

    assert sample.condition_attributes is not None
    assert sample.condition_attributes.text == {"description": "very_good"}
    assert sample.condition_attributes.tensor == {}


def test_hibiki_tokenizer_uses_seconds_for_text_stream_length(tmp_path, monkeypatch):
    patch_audio_read(monkeypatch)
    row = write_manifest_files(tmp_path, duration=0.5)
    tokenizer, interleaver = build_tokenizer(n_q=16, dep_q=8)
    tokenizer.duration_sec = 2.0
    tokenizer.num_audio_frames = math.ceil(2.0 * tokenizer.mimi.frame_rate)

    sample = tokenizer(row, 0.0, tmp_path)

    assert interleaver.last_segment_duration == 0.5
    assert sample.codes.shape[-1] == 25
