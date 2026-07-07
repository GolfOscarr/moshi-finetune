# Hibiki S2ST Fine-Tuning Changes

This document records the local fork changes that add a practical Hibiki
paired source-target speech-to-speech fine-tuning path while preserving the
existing Moshi single-audio fine-tuning path.

## Modified Pipeline

The training entrypoint now supports two data modes:

- `data.mode: moshi` keeps the existing Moshi JSONL format and
  `InterleavedTokenizer` path.
- `data.mode: hibiki_s2st` uses paired source-target JSONL rows and
  `HibikiS2STTokenizer`.

The Hibiki path packs streams as:

```text
[target_text, target_audio_codebooks, source_audio_codebooks]
```

For each chunk, the tokenizer emits:

```text
target_text:  [1, 1, T]
target_audio: [1, dep_q, T]
source_audio: [1, n_q - dep_q, T]
codes:        [1, 1 + n_q, T]
T = ceil(duration_sec * mimi.frame_rate)
```

The existing train and eval loss code is intentionally unchanged. Loss is
computed for target text and the generated `dep_q` target-audio streams. Source
audio streams are teacher-forced conditioning context in this first fork patch.

## Files And Logic

| File | Change |
| --- | --- |
| `train.py` | Selects `InterleavedTokenizer` for Moshi mode or `HibikiS2STTokenizer` for Hibiki mode after model metadata is loaded. |
| `finetune/data/args.py` | Adds `DataArgs.mode` with accepted values `moshi` and `hibiki_s2st`. |
| `finetune/data/data_loader.py` | Passes `data.mode` into dataset construction. |
| `finetune/data/dataset.py` | Preserves the existing `sphn.dataset_jsonl` Moshi path; adds direct paired JSONL row parsing and chunking for Hibiki mode. |
| `finetune/data/hibiki_s2st.py` | Adds paired S2ST schema validation, path resolution, audio chunk encoding, target alignment slicing, stream packing, and speaker-similarity condition normalization. |
| `example/hibiki_1b_s2st.yaml` | Provides a Hibiki 1B LoRA fine-tuning starter config. |
| `example/validate_hibiki_s2st_mock.py` | Verifies the new data path with deterministic fake Mimi/interleaver components and generated scratch data. |
| `tests/test_hibiki_s2st_data.py` | Covers condition normalization, mode validation, 1B/2B stream shapes, condition attributes, and text duration handling. |
| `README.md` | Documents the paired Hibiki JSONL schema and current training-scope limits. |

## Data Schema

Hibiki mode expects one JSON object per line:

```json
{
  "source_audio": "fr/source_000001.wav",
  "target_audio": "en/target_000001.wav",
  "duration": 4.0,
  "target_alignment": "alignments/target_000001.json",
  "speaker_similarity": "good"
}
```

Required fields are `source_audio`, `target_audio`, `duration`, and
`target_alignment`. Relative paths are resolved from the manifest JSONL
directory.

The target alignment JSON contains timestamped target words:

```json
{
  "alignments": [
    ["hello", [0.10, 0.35], "SPEAKER_MAIN"],
    ["world", [0.40, 0.70], "SPEAKER_MAIN"]
  ]
}
```

Speaker similarity labels are normalized to one of:

```text
very_bad, bad, neutral, good, very_good
```

Missing or unknown labels normalize to `neutral` and are returned as:

```python
ConditionAttributes(text={"description": label}, tensor={})
```

## Verification

Recommended focused checks:

```sh
python -m py_compile \
  train.py \
  finetune/data/args.py \
  finetune/data/data_loader.py \
  finetune/data/dataset.py \
  finetune/data/hibiki_s2st.py \
  finetune/data/interleaver.py \
  finetune/eval.py \
  finetune/loss.py

pytest tests/test_hibiki_s2st_data.py -q

python example/validate_hibiki_s2st_mock.py --out-dir ../../tmp/hibiki_s2st_mock
```

Expected mock verifier shape evidence:

```text
ok: 1b codes shape is [1, 17, 13]
ok: 2b codes shape is [1, 33, 13]
ok: condition description is neutral
```

## Known Gaps

This fork change does not implement paper-exact source-stream loss, contextual
alignment, silence insertion, alignment-aware TTS generation, TTS candidate
ranking, Hibiki-M distillation, or ASR-BLEU/speaker-similarity/latency/MOS
benchmark harnesses.
