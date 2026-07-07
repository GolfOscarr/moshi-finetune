# Moshi-Finetune Hibiki S2ST Patch Implementation Plan

## Scope

Implement one small patch for `repos/moshi-finetune` that adds a practical Hibiki paired source-target S2ST fine-tuning path while preserving the current Moshi single-audio conversation path.

Patch artifact to generate:

```text
repos/patches/moshi-finetune-hibiki-s2st-finetuning.patch
```

Use this flat patch path because it is the explicit user request. Local `repos/AGENTS.md` and `repos/patches/AGENTS.md` prefer `repos/patches/<repo-name>/...`; do not switch layouts for this task.

The first patch trains only losses already exposed by the current Moshi model/training loop:

- target text loss
- generated target audio loss over `model.dep_q` streams
- source audio streams as teacher-forced conditioning context

Paper-exact source-stream loss is out of scope. Current `tmp/moshi-src/moshi/moshi/models/lm.py::LMModel.forward()` returns audio logits only for `dep_q` generated audio streams, not all `n_q` streams.

Evidence already checked:

- `repos/moshi-finetune/finetune/data/interleaver.py` currently emits `[text, audio]` from one audio file.
- `repos/moshi-finetune/finetune/data/dataset.py` currently uses `sphn.dataset_jsonl(...)` and expects JSONL rows with `path` and `duration`.
- `repos/moshi-finetune/train.py` computes text loss on `codes[:, :model.audio_offset]` and audio loss on `codes[:, model.audio_offset:model.audio_offset + model.dep_q]`.
- `tmp/moshi-src/moshi/moshi/models/lm.py` asserts `codes.shape[1] == model.num_codebooks == 1 + model.n_q`.
- Hibiki 1B config has `n_q=16`, `dep_q=8`, and `len(delays)=17`.
- Hibiki 2B config has `n_q=32`, `dep_q=16`, and `len(delays)=33`.
- Existing patch collision check found only `repos/patches/AGENTS.md` and `repos/patches/README.md`; no existing patch touches upstream `moshi-finetune` files.

## Files To Change

### `repos/moshi-finetune/finetune/data/args.py`

Add a mode switch to `DataArgs`:

```python
mode: str = "moshi"
```

Add `DataArgs.__post_init__`:

```python
def __post_init__(self) -> None:
    valid_modes = {"moshi", "hibiki_s2st"}
    if self.mode not in valid_modes:
        raise ValueError(
            f"data.mode must be one of {sorted(valid_modes)}, got {self.mode!r}"
        )
```

Keep `train_data`, `eval_data`, and `shuffle` unchanged.

### `repos/moshi-finetune/train.py`

Import the new tokenizer:

```python
from finetune.data.hibiki_s2st import HibikiS2STTokenizer
```

Keep constructing `Interleaver` once, because both modes need target text packing:

```python
interleaver = Interleaver(
    spm,
    mimi.frame_rate,
    model.text_padding_token_id,
    model.end_of_text_padding_id,
    model.zero_token_id,
    keep_main_only=True,
)
```

Replace the unconditional `InterleavedTokenizer(...)` construction with:

```python
if args.data.mode == "moshi":
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )
elif args.data.mode == "hibiki_s2st":
    interleaved_tokenizer = HibikiS2STTokenizer(
        mimi=mimi,
        interleaver=interleaver,
        duration_sec=args.duration_sec,
        n_q=model.n_q,
        dep_q=model.dep_q,
        expected_num_codebooks=model.num_codebooks,
    )
else:
    raise ValueError(f"Unsupported data mode: {args.data.mode}")
```

Do not change train/eval loss slicing in this patch.

### `repos/moshi-finetune/finetune/data/data_loader.py`

Pass `args.mode` to `build_dataset(...)`:

```python
dataset = build_dataset(
    pretrain_data=pretrain_data,
    instruct_tokenizer=instruct_tokenizer,
    seed=seed,
    rank=rank,
    world_size=world_size,
    is_eval=is_eval,
    shuffle_pretrain=args.shuffle,
    mode=args.mode,
)
```

Keep batching and `Batch.collate(...)` unchanged.

### `repos/moshi-finetune/finetune/data/dataset.py`

Keep the current `sphn.dataset_jsonl(...)` path unchanged for `mode == "moshi"`.

Add mode-aware dispatch:

```python
def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: Any,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
    mode: str = "moshi",
) -> Iterator[Sample]:
```

```python
def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: Any,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    mode: str,
) -> Iterator[Sample]:
```

For `mode == "moshi"`:

- preserve current `sphn.dataset_jsonl(...)`
- preserve current `dataset.shuffle(...)` and `.seq(...)`
- yield `instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])`

For `mode == "hibiki_s2st"`:

- iterate JSONL rows directly with `load_file(jsonl_file, rank, world_size)`
- parse each row with `json.loads`
- if `shuffle_at_epoch`, shuffle row order deterministically with the existing `seed` and `rank`
- for each row, emit chunks:

```python
start_sec = 0.0
while start_sec < row["duration"]:
    yield instruct_tokenizer(row, start_sec, jsonl_file.parent)
    start_sec += instruct_tokenizer.duration_sec
```

Keep `parse_data_sources(...)`, `DataDir`, and `DataFile` behavior unchanged.

### `repos/moshi-finetune/README.md`

Add a compact section after the existing dataset preparation section:

- `data.mode: moshi` keeps the current single-audio format.
- `data.mode: hibiki_s2st` expects paired S2ST JSONL.
- Hibiki stream order is `[target_text, target_audio_codebooks, source_audio_codebooks]`.
- First patch objective is target text plus target audio loss.
- Source audio is teacher-forced conditioning context.
- Source-stream loss, contextual alignment, TTS generation, silence insertion, and benchmark metrics are not implemented.

## New Files To Add

### `repos/moshi-finetune/finetune/data/hibiki_s2st.py`

This file owns the paired S2ST schema and tokenizer.

Add:

```python
VALID_SPEAKER_SIMILARITY_LABELS = {
    "very_bad",
    "bad",
    "neutral",
    "good",
    "very_good",
}
```

```python
def normalize_speaker_similarity(label: object) -> str:
    if isinstance(label, str):
        normalized = label.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized in VALID_SPEAKER_SIMILARITY_LABELS:
            return normalized
    return "neutral"
```

```python
def make_condition_attributes(label: object) -> ConditionAttributes:
    return ConditionAttributes(
        text={"description": normalize_speaker_similarity(label)},
        tensor={},
    )
```

```python
def resolve_manifest_path(path: str, manifest_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return manifest_dir / candidate
```

Add `HibikiS2STTokenizer`:

```python
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
```

`__call__` signature:

```python
def __call__(self, row: dict, start_sec: float, manifest_dir: Path) -> Sample:
```

Implementation details:

- Validate required row fields: `source_audio`, `target_audio`, `duration`, `target_alignment`.
- Use existing dependency `sphn`; do not add new heavy dependencies.
- Read source and target chunks with `sphn.read(path, start_sec=start_sec, duration_sec=self.duration_sec)`.
- Normalize waveform shape to mono `[1, samples]` before Mimi encoding. If `sphn.read` returns multiple channels, average channels rather than treating channels as batch items.
- Move tensors to the same device style as the current tokenizer. Existing code uses CUDA directly; for testability, prefer:

```python
device = getattr(self.interleaver, "device", "cuda")
audio_tensor = torch.as_tensor(wav, dtype=torch.float32, device=device)
```

- Encode with `self.mimi.encode(audio_tensor[:, None])`, producing `[1, q, T_actual]` for mono input.
- Pad or truncate encoded tokens to `self.num_audio_frames` with `self.interleaver.zero_padding`.
- Load target alignment JSON from `target_alignment`.
- Slice alignments to `[start_sec, start_sec + duration_sec]` and subtract `start_sec` from timestamps.
- Build target text with a duration in seconds, not frame count:

```python
segment_duration_sec = min(self.duration_sec, max(0.0, row["duration"] - start_sec))
target_text = self.interleaver.prepare_item(alignments, segment_duration_sec)
```

- Pad or truncate `target_text` to `self.num_audio_frames`.
- Select target/source streams:

```python
target_audio = target_audio_tokens[:, : self.dep_q, :]
source_audio = source_audio_tokens[:, : self.n_q - self.dep_q, :]
```

- Concatenate:

```python
codes = torch.cat([target_text, target_audio, source_audio], dim=1)
```

- Validate:

```python
if codes.shape[1] != self.expected_num_codebooks:
    raise ValueError(
        f"Expected {self.expected_num_codebooks} streams, got {codes.shape[1]}"
    )
```

- Return:

```python
return Sample(codes, make_condition_attributes(row.get("speaker_similarity")))
```

### `repos/moshi-finetune/example/hibiki_1b_s2st.yaml`

Add:

```yaml
data:
  mode: hibiki_s2st
  eval_data: ''
  shuffle: true
  train_data: ''

moshi_paths:
  hf_repo_id: "kyutai/hibiki-1b-pytorch-bf16"

full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.
  ft_embed: false

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

duration_sec: 40
batch_size: 1
max_steps: 2000
gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

seed: 0
log_freq: 1
eval_freq: 100
do_eval: false
do_ckpt: true
ckpt_freq: 100
save_adapters: true
run_dir: ""
```

### `repos/moshi-finetune/tests/test_hibiki_s2st_data.py`

Add focused pytest coverage using fake Mimi/interleaver/tokenizer inputs where possible. Do not download Hibiki checkpoints in tests.

Required tests:

- `test_normalize_speaker_similarity_accepts_valid_labels`
- `test_normalize_speaker_similarity_maps_missing_and_unknown_to_neutral`
- `test_data_args_default_mode_is_moshi`
- `test_data_args_rejects_unknown_mode`
- `test_hibiki_tokenizer_emits_17_streams_for_1b_shape`
- `test_hibiki_tokenizer_emits_33_streams_for_2b_shape`
- `test_hibiki_tokenizer_returns_description_condition_attributes`
- `test_hibiki_tokenizer_uses_seconds_for_text_stream_length`

Use a fake Mimi shaped like:

```python
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
```

For unit tests, monkeypatch `sphn.read` to return deterministic NumPy arrays instead of writing real audio files.

### `repos/moshi-finetune/example/validate_hibiki_s2st_mock.py`

Add a lightweight validation script that writes generated artifacts outside Git. It should accept:

```text
--out-dir
```

Default for this project should be the parent repo scratch path when run from the parent:

```sh
python repos/moshi-finetune/example/validate_hibiki_s2st_mock.py --out-dir tmp/hibiki_s2st_mock
```

The script should:

- create tiny source and target waveforms
- create target alignment JSON
- create a paired JSONL manifest
- instantiate the new tokenizer with fake Mimi for 1B and 2B shape checks
- verify stream layout and condition attributes

## Data Schema

Hibiki S2ST JSONL row:

```json
{
  "source_audio": "fr/source_000001.wav",
  "target_audio": "en/target_000001.wav",
  "duration": 4.0,
  "target_alignment": "alignments/target_000001.json",
  "speaker_similarity": "good",
  "source_language": "fr",
  "target_language": "en",
  "split": "train",
  "dataset_source": "mock"
}
```

Required:

- `source_audio`
- `target_audio`
- `duration`
- `target_alignment`

Optional:

- `speaker_similarity`
- `source_language`
- `target_language`
- `split`
- `dataset_source`
- `source_transcript`
- `target_text`
- `alignment_method`
- `speaker_similarity_score`

Target alignment JSON:

```json
{
  "alignments": [
    ["hello", [0.10, 0.35], "SPEAKER_MAIN"],
    ["world", [0.40, 0.70], "SPEAKER_MAIN"]
  ]
}
```

Path resolution:

- absolute paths are used as-is
- relative paths are resolved against the manifest JSONL directory

## Training/Data Flow

1. `TrainArgs.load(...)` loads `data.mode`.
2. `train.py` loads checkpoint metadata with `loaders.CheckpointInfo.from_hf_repo(...)`.
3. `train.py` freezes Mimi as before.
4. `train.py` builds the model as before.
5. `train.py` builds `Interleaver` as before for target text alignment.
6. `data.mode == "moshi"` uses current `InterleavedTokenizer`.
7. `data.mode == "hibiki_s2st"` uses `HibikiS2STTokenizer`.
8. `build_data_loader(...)` passes `args.mode` into `build_dataset(...)`.
9. `build_dataset(...)` preserves current Moshi behavior for default mode.
10. Hibiki mode reads paired rows, chunks each row by `duration_sec`, and tokenizes each chunk.
11. The tokenizer reads source and target audio chunks.
12. The tokenizer encodes source and target audio with frozen Mimi.
13. The tokenizer loads target timestamps and builds target text at Mimi frame rate.
14. The tokenizer packs `codes`.
15. Existing train/eval loops compute target text and target audio loss.

## Shape Contracts

For `T = ceil(duration_sec * mimi.frame_rate)`:

```text
target_text:  [1, 1, T]
target_audio: [1, dep_q, T]
source_audio: [1, n_q - dep_q, T]
codes:        [1, 1 + n_q, T]
```

Hibiki 1B / Hibiki-M:

```text
n_q = 16
dep_q = 8
K = 17
stream 0     target text
streams 1-8 target audio
streams 9-16 source audio
```

Hibiki 2B:

```text
n_q = 32
dep_q = 16
K = 33
stream 0      target text
streams 1-16  target audio
streams 17-32 source audio
```

Loss target mapping remains:

```text
text loss target:  codes[:, :1]
audio loss target: codes[:, 1:1 + dep_q]
source streams:    codes[:, 1 + dep_q:]
```

Padding:

- encoded audio tail padding uses `zero_token_id`
- text stream keeps existing `text_padding_token_id` and `end_of_text_padding_id` behavior
- absent frames outside the sample duration use `zero_token_id`

## Compatibility Strategy

- Default `data.mode` is `"moshi"`.
- Existing JSONL rows with `path` and `duration` continue to work.
- Existing `InterleavedTokenizer` behavior remains unchanged.
- Existing train/eval objective remains unchanged.
- New code is routed only through `data.mode: hibiki_s2st`.
- New Hibiki code uses existing dependencies: `torch`, `sphn`, `json`, `pathlib`, `moshi.conditioners.ConditionAttributes`.
- No checkpoints, datasets, generated audio, or binary artifacts are committed.

## Tests And Validation

Run syntax checks:

```sh
cd repos/moshi-finetune
python -m py_compile \
  train.py \
  finetune/data/args.py \
  finetune/data/data_loader.py \
  finetune/data/dataset.py \
  finetune/data/hibiki_s2st.py \
  finetune/data/interleaver.py \
  finetune/eval.py \
  finetune/loss.py
```

Run focused tests:

```sh
cd repos/moshi-finetune
pytest tests/test_hibiki_s2st_data.py -q
```

Run mock verifier from the parent repo:

```sh
python repos/moshi-finetune/example/validate_hibiki_s2st_mock.py --out-dir tmp/hibiki_s2st_mock
```

Expected verifier output should include:

```text
ok: wrote mock data under tmp/hibiki_s2st_mock
ok: 1b codes shape is [1, 17, T]
ok: 2b codes shape is [1, 33, T]
ok: condition description is neutral
```

## Mock-Data Verification

The mock verifier must exercise the actual new Hibiki path without committing generated files.

It should write:

```text
tmp/hibiki_s2st_mock/source.wav
tmp/hibiki_s2st_mock/target.wav
tmp/hibiki_s2st_mock/target_alignment.json
tmp/hibiki_s2st_mock/manifest.jsonl
```

Manifest row:

```json
{"source_audio":"source.wav","target_audio":"target.wav","duration":1.0,"target_alignment":"target_alignment.json","speaker_similarity":"unknown"}
```

Alignment:

```json
{"alignments":[["hello",[0.10,0.30],"SPEAKER_MAIN"],["world",[0.35,0.55],"SPEAKER_MAIN"]]}
```

Assertions:

- `codes.shape[0] == 1`
- `codes.shape[1] == 17` for 1B settings
- `codes.shape[1] == 33` for 2B settings
- `codes.shape[2] == ceil(duration_sec * frame_rate)`
- stream `0` is target text
- streams `1:1 + dep_q` are target audio
- streams `1 + dep_q:` are source audio
- condition is `ConditionAttributes(text={"description": "neutral"}, tensor={})`

## Patch-Generation Steps

1. Re-check patch collision:

```sh
find repos/patches -maxdepth 2 -type f -print | sort
```

2. Modify only planned files under `repos/moshi-finetune`.

3. Run syntax checks, tests, and mock verifier.

4. Generate patch:

```sh
git -C repos/moshi-finetune diff > repos/patches/moshi-finetune-hibiki-s2st-finetuning.patch
```

5. Verify patch applies from the submodule root:

```sh
git -C repos/moshi-finetune apply --check ../patches/moshi-finetune-hibiki-s2st-finetuning.patch
```

6. Verify patch is non-empty:

```sh
test -s repos/patches/moshi-finetune-hibiki-s2st-finetuning.patch
```

## Risks And Open Questions

- Mock validation proves routing, schema, packing, shape contracts, and condition handling; it does not prove real Hibiki training quality.
- Real Hibiki training requires checkpoint access, GPU memory, and real paired data.
- The patch deliberately omits source-stream loss because current model output does not expose logits for all source streams.
- The patch omits MADLAD-3B translation, contextual alignment, silence insertion, alignment-aware TTS, TTS candidate ranking, source-audio augmentation, Hibiki-M distillation, and benchmark metrics.
- Speaker-similarity labels default to `neutral`; this is practical but not paper-faithful quantile labeling.
- Source EOS insertion is not part of this first patch unless it can be added from an optional manifest field without changing model logic or increasing scope.

## Stop Conditions

Implementation is complete only when:

- `repos/patches/moshi-finetune-hibiki-s2st-finetuning.patch` exists.
- The patch contains only reviewable changes to the planned upstream files.
- Current Moshi mode still defaults to existing behavior.
- Hibiki mode produces `K=17` for 1B and `K=33` for 2B in mock validation.
- Condition labels normalize to the five Hibiki-supported values.
- Syntax checks pass.
- Focused tests pass.
- Mock verifier passes.
- `git -C repos/moshi-finetune apply --check ../patches/moshi-finetune-hibiki-s2st-finetuning.patch` passes.
