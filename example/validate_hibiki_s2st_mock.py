import argparse
import json
import math
import sys
import types
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ConditionAttributes:
    text: dict = field(default_factory=dict)
    tensor: dict = field(default_factory=dict)


moshi_module = types.ModuleType("moshi")
conditioners_module = types.ModuleType("moshi.conditioners")
conditioners_module.ConditionAttributes = ConditionAttributes
sys.modules["moshi"] = moshi_module
sys.modules["moshi.conditioners"] = conditioners_module

from finetune.data.hibiki_s2st import HibikiS2STTokenizer  # noqa: E402
import finetune.data.hibiki_s2st as hibiki_s2st  # noqa: E402


class FakeMimi:
    sample_rate = 24000
    frame_rate = 12.5

    def __init__(self, num_codebooks: int):
        self.num_codebooks = num_codebooks

    def encode(self, audio):
        batch = audio.shape[0]
        frames = max(1, math.ceil(audio.shape[-1] / 1920))
        offset = int(audio.mean().item() * 1000)
        values = torch.arange(
            self.num_codebooks * frames,
            device=audio.device,
            dtype=torch.long,
        )
        return (values + offset).view(1, self.num_codebooks, frames).repeat(
            batch, 1, 1
        )


class FakeInterleaver:
    device = "cpu"
    zero_padding = -1

    def prepare_item(self, alignments, segment_duration):
        frames = math.ceil(segment_duration * FakeMimi.frame_rate)
        return torch.full((1, 1, frames), 101, dtype=torch.long)


def write_wav(path: Path, value: int) -> None:
    samples = np.full(24000, value, dtype=np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(24000)
        f.writeframes(samples.tobytes())


def write_mock_data(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / "source.wav", 0)
    write_wav(out_dir / "target.wav", 1000)

    (out_dir / "target_alignment.json").write_text(
        json.dumps(
            {
                "alignments": [
                    ["hello", [0.10, 0.30], "SPEAKER_MAIN"],
                    ["world", [0.35, 0.55], "SPEAKER_MAIN"],
                ]
            }
        )
    )
    row = {
        "source_audio": "source.wav",
        "target_audio": "target.wav",
        "duration": 1.0,
        "target_alignment": "target_alignment.json",
        "speaker_similarity": "unknown",
    }
    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text(json.dumps(row) + "\n")
    return manifest_path


def fake_read(path, *args, **kwargs):
    value = 1.0 if Path(path).name == "target.wav" else 0.0
    return np.full((1, 24000), value, dtype=np.float32), 24000


def build_sample(row: dict, manifest_dir: Path, n_q: int, dep_q: int):
    tokenizer = HibikiS2STTokenizer(
        mimi=FakeMimi(num_codebooks=dep_q),
        interleaver=FakeInterleaver(),
        duration_sec=1.0,
        n_q=n_q,
        dep_q=dep_q,
        expected_num_codebooks=1 + n_q,
    )
    return tokenizer(row, 0.0, manifest_dir)


def assert_layout(sample, n_q: int, dep_q: int) -> None:
    expected_t = math.ceil(FakeMimi.frame_rate)
    assert sample.codes.shape == (1, 1 + n_q, expected_t)
    assert torch.all(sample.codes[:, 0] == 101)
    assert torch.all(sample.codes[:, 1 : 1 + dep_q] >= 1000)
    assert torch.all(sample.codes[:, 1 + dep_q :] < 1000)
    assert sample.condition_attributes is not None
    assert sample.condition_attributes.text == {"description": "neutral"}
    assert sample.condition_attributes.tensor == {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tmp/hibiki_s2st_mock")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = write_mock_data(out_dir)
    row = json.loads(manifest_path.read_text().splitlines()[0])

    hibiki_s2st.sphn.read = fake_read
    sample_1b = build_sample(row, manifest_path.parent, n_q=16, dep_q=8)
    sample_2b = build_sample(row, manifest_path.parent, n_q=32, dep_q=16)
    assert_layout(sample_1b, n_q=16, dep_q=8)
    assert_layout(sample_2b, n_q=32, dep_q=16)

    print(f"ok: wrote mock data under {out_dir}")
    print(f"ok: 1b codes shape is {list(sample_1b.codes.shape)}")
    print(f"ok: 2b codes shape is {list(sample_2b.codes.shape)}")
    print(
        "ok: condition description is "
        f"{sample_1b.condition_attributes.text['description']}"
    )


if __name__ == "__main__":
    main()
