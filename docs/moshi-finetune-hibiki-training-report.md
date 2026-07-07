# Moshi-Finetune Code Structure And Hibiki Training Adaptation Report

Date: 2026-07-05

## Scope

This report analyzes `repos/moshi-finetune`, the official Kyutai Moshi fine-tuning repository, as a candidate scaffold for Hibiki speech-to-speech translation training.

Paper and official reference metadata:

- Paper title: High-Fidelity Simultaneous Speech-To-Speech Translation
- arXiv URL: https://arxiv.org/abs/2502.03382
- Authors: Tom Labiausse, Laurent Mazare, Edouard Grave, Patrick Perez, Alexandre Defossez, Neil Zeghidour
- Year: 2025
- Official Hibiki GitHub repo: https://github.com/kyutai-labs/hibiki
- Official Hibiki project page: https://huggingface.co/spaces/kyutai/hibiki-samples
- Official Moshi fine-tuning repo analyzed here: https://github.com/kyutai-labs/moshi-finetune

Local references used:

- `repos/moshi-finetune/`
- `papers/S2ST/Hibiki.pdf`
- `papers/reviews/Hibiki.md`
- `models/hibiki/architecture-analysis.md`
- `models/hibiki/hibiki-1b-pytorch-bf16/config.json`
- `models/hibiki/hibiki-2b-pytorch-bf16/config.json`
- `tmp/moshi-src/moshi/moshi/models/loaders.py`
- `tmp/moshi-src/moshi/moshi/models/lm.py`
- `tmp/moshi-src/moshi/moshi/run_inference.py`

This is a read-only architecture report. It does not propose patch contents or modify the upstream submodule.

## Executive Finding

`moshi-finetune` is a useful training scaffold for Hibiki, but it is not a complete Hibiki S2ST training recipe.

The reusable part is the model-training infrastructure: Moshi/Hibiki checkpoint loading, Mimi freezing, LoRA insertion, FSDP wrapping, masked token loss, mixed precision, optimizer scheduling, and checkpoint export.

The non-reusable part as-is is the dataset/tokenization contract. Current `moshi-finetune` builds one text stream plus one audio stream group from a single audio file. Hibiki training requires one generated target-text stream, generated target-audio codebooks, and source/input audio codebooks in the same multistream language-model tensor.

## Paper Details From Fresh PDF Read

The Hibiki paper describes a staged French-to-English simultaneous S2ST recipe, not only a single fine-tuning run.

### Model And Tokenization Details

- Hibiki is a decoder-only model that receives source speech and generates translated speech using a multistream architecture.
- It jointly produces target-language text tokens and target-speech audio tokens.
- It uses Mimi, a causal streaming codec, to encode source waveform `X` and target waveform `Y` into low-frame-rate discrete tokens.
- Mimi runs at 24 kHz waveform sample rate and 12.5 Hz token frame rate.
- Mimi supports up to 32 RVQ codebooks, but Hibiki uses at most 16 codebooks per stream.
- Codebook 1 is treated as semantic; codebooks 2 and later are acoustic and arranged coarse-to-fine.
- Hibiki concatenates delayed target audio tokens and delayed source audio tokens along the codebook axis.
- Source audio tokens are modeled during training, but at inference their predictions are skipped and actual source tokens are supplied.
- Hibiki also predicts an aligned target text stream, the Inner Monologue, at the same frame rate.
- The Temporal Transformer consumes previous time-step target audio, source audio, and text tokens through dedicated embeddings whose contributions are summed.
- The Depth Transformer runs for `2 * Q` steps in the paper description: first for output/target-stream codebooks, then for input/source-stream codebooks.
- Acoustic codebooks use a delay of 2 codec frames; the delay is removed before codec decoding.

### Alignment And Synthetic Data Details

- The paper creates synthetic parallel data by translating and resynthesizing transcripts of single-language French audio.
- Contextual alignment estimates which source word a target word should wait for by measuring next-target-word log-likelihood gains under MADLAD-3B as the source prefix grows.
- Audio-domain alignment transcribes source and target with Whisper timestamps, applies contextual alignment, requires target speech to lag by at least 2 seconds, and smooths spikes higher than 25% of average delay over a 5-word window.
- Speech translation training uses silence insertion to force simultaneous target speech when target words would otherwise appear too early.
- Fine-tuning uses alignment-aware TTS instead of simple silence insertion.
- Alignment-aware TTS constrains the generated text stream exactly, allows only padding-token timing freedom, delays audio behind text so audio is conditioned on text, forces padding if speech is early, and penalizes padding logits from 0 to -2 as lag increases from 1 to 2 seconds.
- The paper samples 6 to 8 TTS generations per input and selects by word error rate first, speaker similarity second.
- Alignment-aware TTS is used only for the fine-tuning speech translation dataset.

### Voice Transfer Details

- CVSS-T is treated as a standard S2ST voice-transfer training set, but its source-target speaker similarity is low.
- The paper reports CVSS-T average speaker similarity around 0.23 and resynthesized CVSS-T around 0.47.
- Training examples receive one discrete voice-transfer label: `very bad`, `bad`, `neutral`, `good`, or `very good`.
- Labels are assigned from speaker-similarity quantiles.
- Quantiles are computed before combining synthetic data with CVSS-T to prevent the condition from becoming a dataset identifier.
- Each label has a learnable embedding added to model inputs at every time step.
- Inference always uses the `very good` label.
- Classifier-free guidance computes logits with `very_good` and `very_bad` conditions and samples with a guidance coefficient.

### Training Protocol Details

All stages use AdamW, cosine learning-rate scheduling, weight decay 0.1, and betas `(0.9, 0.95)`.

| Stage | Paper detail |
| --- | --- |
| Text pretraining | Train the Temporal Transformer from scratch on multilingual text-only next-token prediction for 600K steps. Batch is 1,024 sequences of length 4,096. Warmup is 2K steps. Max LR is 4.8e-4. Data is filtered Common Crawl plus curated sources such as Wikipedia, StackExchange, and scientific articles, with 12.5% multilingual documents. |
| Audio pretraining | Start from the pretrained text model. Train on non-parallel French and English audio with a single stream. Train for 1,450K steps, batch size 144, LR 2e-4. Then duplicate Depth Transformer weights for multistream modeling. |
| Speech translation training | Build about 40K hours in each language. Start from expressive French audio. Extract about 2.3M single-speaker utterances around 60 seconds each. Transcribe with Whisper large-v3. Segment with PySBD. Translate with MADLAD-3B. Synthesize English target speech with TTS conditioned on a 10-second utterance from the original French speaker. Apply silence insertion. Train for 150K steps, batch size 96, LR 3e-5. Compute loss on both source and target streams. Use speaker-similarity conditional training, source-audio noise augmentation, source-audio EOS at the first frame after source speech ends, and text EOS for model speech end. |
| Speech translation fine-tuning | Build close to 900 hours from alignment-aware TTS generations: long-form utterances plus improved CVSS-T/train with natural pauses and high speaker similarity. Fine-tune for 8K steps, batch size 8, LR 2e-6. Use speaker-similarity conditional training, special EOS tokens, and loss on both streams. |
| Hibiki-M training | Use the same text and audio pretraining stages. During speech translation training, soft-distill from Hibiki. Then run the same fine-tuning step without distillation. |

### Evaluation And Inference Details Relevant To Training

- Short-form evaluation is CVSS-C Fr-En test; 99% of sequences are shorter than 10 seconds.
- Long-form evaluation is Audio-NTREX, 10 hours of speech in each language, 10 speakers, average utterance length around 50 seconds.
- Real interpretation comparison uses 90 VoxPopuli European Parliament interpretation samples.
- Metrics include BLEU, ASR-BLEU, cross-lingual speaker similarity, End Offset, LAAL, and MOS for quality, speaker similarity, and naturalness.
- Inference encodes source audio with Mimi, feeds source tokens to Hibiki, and decodes generated output tokens.
- At source end, inference sends EOS to the model and continues sampling until the model emits its own EOS.
- Inference parameters are cross-validated using held-out 8% Audio-NTREX and CVSS-C validation.
- Paper default for Audio-NTREX is guidance coefficient 3.0, temperature 0.8, top-k 250 for audio, and top-k 50 for text.
- CVSS uses the same settings except text temperature is 0.1.

## Ranked Synthesis

| Rank | Finding | Confidence | Basis |
| --- | --- | --- | --- |
| 1 | `moshi-finetune` can probably instantiate and train Hibiki checkpoints with limited model-loading changes. | High | The fine-tune repo calls Moshi `CheckpointInfo` and `LMModel`; local Moshi loader parses `model_type: "hibiki"` and handles Hibiki-specific EOS behavior. |
| 2 | The current training loop is reusable for LoRA-style Hibiki adaptation once `codes` are packed correctly. | High | The loop forwards `model(codes=...)`, computes text loss and `dep_q` audio loss, and is otherwise architecture-generic. |
| 3 | The current data loader/interleaver is not sufficient for Hibiki S2ST. | High | It expects JSONL rows with one audio path and sidecar word timestamps, then emits `[text_tokens, audio_tokens]`. |
| 4 | A first practical Hibiki fine-tune path should focus on target text plus target audio loss while source audio is teacher-forced as conditioning input. | Medium | This matches the current `LMModel.forward()` output shape, but the Hibiki paper says training computes loss on both source and target streams. |
| 5 | Reproducing paper-exact Hibiki training is blocked more by data construction than by model architecture. | High | The paper review records undisclosed synthetic corpora, incomplete source-audio augmentation details, and undisclosed exact per-stream loss weights. |

## Repository Structure

`repos/moshi-finetune` is small and focused.

| Path | Role |
| --- | --- |
| `README.md` | User guide for Moshi LoRA fine-tuning, dataset preparation, training commands, and inference. |
| `pyproject.toml` | Python package metadata and dependencies. The project depends on `moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi`. |
| `example/moshi_7B.yaml` | Main example training config. It points to `kyutai/moshiko-pytorch-bf16` by default. |
| `train.py` | Main training entrypoint. Loads config, model, tokenizer, Mimi, data loaders, optimizer, metrics, and checkpointer. |
| `annotate.py` | Whisper timestamp annotation helper for sidecar `.json` transcript files. |
| `finetune/args.py` | Dataclass config for model paths, LoRA, optimizer, logging, precision, and checkpoint behavior. |
| `finetune/data/args.py` | Minimal data config: `train_data`, `eval_data`, and `shuffle`. |
| `finetune/data/dataset.py` | JSONL/data-source parsing, sharding, chunking, and iterator construction. |
| `finetune/data/interleaver.py` | Converts waveform chunks and sidecar alignments into LM token/code tensors. |
| `finetune/data/data_loader.py` | Batches `Sample` objects into `Batch` objects. |
| `finetune/wrapped_model.py` | Builds Moshi `LMModel`, injects LoRA, freezes non-LoRA params, and wraps in FSDP. |
| `finetune/loss.py` | Masked cross-entropy for text and audio tokens. |
| `finetune/eval.py` | Evaluation loop over the same token-loss objective. |
| `finetune/checkpointing.py` | Saves LoRA adapters or merged model checkpoints as safetensors. |
| `finetune/mixed_precision.py` | Upcast/downcast helpers for optimizer and train-time parameter dtype. |
| `finetune/distributed.py` | Torch distributed helpers. |
| `finetune/monitoring/` | TensorBoard and optional W&B metric logging. |

## Training Flow

The training path is:

1. `train.py` loads YAML into `TrainArgs`.
2. `loaders.CheckpointInfo.from_hf_repo(...)` resolves model config, LM weights, Mimi weights, and tokenizer.
3. Mimi is loaded, set to eval mode, and frozen.
4. `get_fsdp_model(...)` creates `LMModel` on `meta`, loads safetensors, inserts LoRA if enabled, freezes non-trainable parameters, then wraps with FSDP for multi-GPU.
5. `InterleavedTokenizer` combines a frozen Mimi encoder with an `Interleaver`.
6. `build_data_loader(...)` yields batches of `codes`.
7. The training loop calls `output = model(codes=codes, condition_tensors=condition_tensors)`.
8. The loop computes:
   - text loss against `codes[:, :model.audio_offset]`
   - audio loss against `codes[:, model.audio_offset:model.audio_offset + model.dep_q]`
9. The optimizer is AdamW with betas `(0.9, 0.95)`, weight decay from config, and `OneCycleLR`.
10. Checkpointing saves LoRA-only adapters or merged consolidated weights.

The core loop is already close to a generic Moshi-family language-model fine-tuning loop.

## Paper Training Details Mapped To `moshi-finetune`

This table separates what the paper requires from what the fine-tuning repository already implements.

| Paper detail | Applied in `moshi-finetune`? | Repo location | Notes for Hibiki |
| --- | --- | --- | --- |
| Load Moshi-family LM config, LM weights, Mimi weights, and tokenizer | Yes | `repos/moshi-finetune/train.py:126-145`, `repos/moshi-finetune/finetune/wrapped_model.py:121-150`, `tmp/moshi-src/moshi/moshi/models/loaders.py:180-313` | Loader can parse Hibiki config because local Moshi supports `model_type: "hibiki"`. |
| Freeze Mimi codec during LM fine-tuning | Yes | `repos/moshi-finetune/train.py:145-148` | Matches Hibiki use of pretrained Mimi as a fixed codec during S2ST model training. |
| Use Mimi token frame rate for text/audio alignment | Yes, for current single-audio path | `repos/moshi-finetune/train.py:155-165`, `repos/moshi-finetune/finetune/data/interleaver.py:171-210`, `repos/moshi-finetune/finetune/data/interleaver.py:247-288` | Reusable for target-text stream construction, but must be extended to source-target audio pairs. |
| Build aligned text stream with padding and end-of-padding token | Yes | `repos/moshi-finetune/finetune/data/interleaver.py:171-210` | This is the closest existing implementation to Hibiki Inner Monologue text packing. |
| Keep main speaker alignments only | Yes | `repos/moshi-finetune/train.py:155-162`, `repos/moshi-finetune/finetune/data/interleaver.py:127-130`, `repos/moshi-finetune/finetune/data/interleaver.py:222-225` | Useful for target-side text alignment if target side has speaker tags. |
| Encode one audio chunk with Mimi | Yes | `repos/moshi-finetune/finetune/data/interleaver.py:254-265` | Needs replacement or extension to encode both source and target audio. |
| Concatenate text tokens and audio tokens into `codes` | Yes, but Moshi-shaped | `repos/moshi-finetune/finetune/data/interleaver.py:288` | Current shape is `[text, one audio group]`; Hibiki needs `[target text, target audio, source audio]`. |
| Train with `codes` tensor shape `[B, K, T]` | Yes | `repos/moshi-finetune/train.py:244-255`, `tmp/moshi-src/moshi/moshi/models/lm.py:322-347` | This part is compatible with Hibiki if `K` equals `1 + n_q`. |
| Use model delay pattern to align logits and masks | Yes, in Moshi dependency | `tmp/moshi-src/moshi/moshi/models/lm.py:348-377` | Hibiki checkpoint `delays` encode target/source stream delays. |
| Text loss on generated text stream | Yes | `repos/moshi-finetune/train.py:256-266`, `repos/moshi-finetune/finetune/loss.py:5-31`, `repos/moshi-finetune/finetune/eval.py:48-58` | Maps to target-text Inner Monologue loss. |
| Audio loss on generated target codebooks | Yes | `repos/moshi-finetune/train.py:267-273`, `repos/moshi-finetune/finetune/loss.py:5-31`, `repos/moshi-finetune/finetune/eval.py:59-65` | The existing slice uses `model.dep_q`, which maps to generated target codebooks. |
| Extra weight on first/semantic audio codebook | Yes | `repos/moshi-finetune/finetune/loss.py:16-18`, `repos/moshi-finetune/example/moshi_7B.yaml:18` | Paper says first codebook carries semantic information; repo already supports upweighting it. |
| Downweight text padding | Yes | `repos/moshi-finetune/train.py:260-265`, `repos/moshi-finetune/finetune/loss.py:19-22`, `repos/moshi-finetune/example/moshi_7B.yaml:19` | Relevant because Hibiki text stream is sparse at 12.5 Hz. |
| Loss on both source and target streams | Partially/no | `repos/moshi-finetune/train.py:267-273`, `tmp/moshi-src/moshi/moshi/models/lm.py:371-374`, `tmp/moshi-src/moshi/moshi/models/lm.py:410-448` | Current training loss targets only `dep_q` audio streams. Paper says loss is computed on both source and target streams. Supporting source-stream loss likely requires deeper LM/output changes. |
| AdamW optimizer | Yes | `repos/moshi-finetune/train.py:196-203` | Betas `(0.9, 0.95)` and weight decay are paper-matching. |
| Weight decay 0.1 | Yes | `repos/moshi-finetune/finetune/args.py:23-27`, `repos/moshi-finetune/train.py:196-203`, `repos/moshi-finetune/example/moshi_7B.yaml:26-29` | Config default and example match paper value. |
| Momentum/betas `(0.9, 0.95)` | Yes | `repos/moshi-finetune/train.py:196-203` | Direct paper match. |
| Cosine learning-rate schedule | Approximate, not exact | `repos/moshi-finetune/train.py:205-210`, `repos/moshi-finetune/finetune/args.py:23-27` | Repo uses `OneCycleLR` with `pct_start`, not a plain cosine schedule with explicit warmup steps. |
| Text pretraining stage | No | No location | Repo fine-tunes an existing checkpoint; it does not train Temporal Transformer from scratch on text. |
| Audio pretraining stage | No | No location | Repo does not train the single-stream audio-pretraining stage or duplicate Depth Transformer weights. |
| Speech translation training data construction from French expressive audio | No | No location | Repo only accepts prepared audio JSONL plus sidecar alignments. |
| Whisper transcription helper | Yes, but not Hibiki paper-exact | `repos/moshi-finetune/annotate.py:73-147`, `repos/moshi-finetune/annotate.py:201-246` | Repo uses `whisper_timestamped` and defaults to `medium`; paper uses Whisper large-v3 for translation-data transcription. |
| PySBD sentence segmentation | No | No location | Required for paper-style transcript segmentation before MADLAD translation. |
| MADLAD-3B translation | No | No location | Required for contextual alignment and synthetic target-text generation. |
| Contextual alignment from MT log-likelihood deltas | No | No location | Core paper contribution absent from `moshi-finetune`. |
| Require target lag of at least 2 seconds and smooth alignment spikes | No | No location | Needs separate data-construction pipeline. |
| Silence insertion for speech translation training | No | No location | Needs audio editing or manifest-level target timing construction. |
| Alignment-aware TTS for fine-tuning data | No | No location | Repo can consume generated target audio, but cannot generate it. |
| 6 to 8 TTS generations and selection by WER then speaker similarity | No | No location | Needs separate TTS generation and ranking pipeline. |
| Speaker-similarity condition labels | Partially | `repos/moshi-finetune/train.py:248-252`, `repos/moshi-finetune/finetune/data/interleaver.py:288`, `models/hibiki/hibiki-1b-pytorch-bf16/config.json:44-69`, `models/hibiki/hibiki-2b-pytorch-bf16/config.json:61-86` | Plumbing exists if samples return `ConditionAttributes`; current data path returns raw JSON `text_conditions`, so Hibiki labels need normalization. |
| Inference-time `very_good` condition and `very_bad` negative condition for CFG | Yes in Moshi inference, not training repo | `tmp/moshi-src/moshi/moshi/run_inference.py:34-57` | Useful for validating adapters after training. |
| Source-audio noise augmentation | No | No location | Paper mentions it but does not disclose exact recipe. |
| Source EOS token at first frame after source speech ends | No in training data path; yes in Moshi inference | `tmp/moshi-src/moshi/moshi/run_inference.py:137-160` | Training tokenizer must add source EOS to source audio streams. |
| Text EOS token to indicate end of model speech | Not paper-specific | `repos/moshi-finetune/finetune/data/interleaver.py:148-169` supports turn EOS, but current training setup uses `keep_main_only=True` | Hibiki tokenizer should explicitly place model-speech EOS in target text stream. |
| Hibiki-M soft distillation from Hibiki | No | No location | Requires teacher model outputs/loss; absent from repo. |
| LoRA fine-tuning | Yes, but not in Hibiki paper | `repos/moshi-finetune/finetune/args.py:10-20`, `repos/moshi-finetune/finetune/wrapped_model.py:121-150`, `repos/moshi-finetune/finetune/wrapped_model.py:173-184`, `repos/moshi-finetune/example/moshi_7B.yaml:11-16` | Practical adaptation mechanism supplied by repo, not paper training protocol. |
| Full fine-tuning option | Yes | `repos/moshi-finetune/train.py:93-97`, `repos/moshi-finetune/finetune/args.py:107-113`, `repos/moshi-finetune/finetune/wrapped_model.py:173-184` | Useful if resources allow paper-closer fine-tuning. |
| Save LoRA adapter or merged model | Yes | `repos/moshi-finetune/finetune/checkpointing.py:54-59`, `repos/moshi-finetune/finetune/checkpointing.py:97-246`, `repos/moshi-finetune/example/moshi_7B.yaml:40` | Reusable for adapter experiments. |
| Evaluation loss loop | Yes, but LM-loss only | `repos/moshi-finetune/finetune/eval.py:24-88` | Paper metrics such as ASR-BLEU, speaker similarity, End Offset, LAAL, and MOS are absent. |

## Applied Training Settings Already Present In Repo

The following settings are already implemented in `moshi-finetune` and can be carried into Hibiki adaptation:

- **AdamW with paper betas**: `repos/moshi-finetune/train.py:196-203` uses `AdamW(..., betas=(0.9, 0.95), weight_decay=args.optim.weight_decay)`.
- **Weight decay 0.1 default**: `repos/moshi-finetune/finetune/args.py:23-27` defaults `weight_decay` to `0.1`; `repos/moshi-finetune/example/moshi_7B.yaml:26-29` also sets `weight_decay: 0.1`.
- **Fine-tuning LR 2e-6 in example config**: `repos/moshi-finetune/example/moshi_7B.yaml:26-29` sets `lr: 2e-6`, matching Hibiki paper fine-tuning LR.
- **Gradient clipping**: `repos/moshi-finetune/finetune/args.py:81-84` defines `max_norm`; `repos/moshi-finetune/train.py:299-300` applies `clip_grad_norm_`.
- **Frozen Mimi**: `repos/moshi-finetune/train.py:145-148` freezes Mimi parameters.
- **LoRA or full fine-tuning switch**: `repos/moshi-finetune/train.py:93-97` enforces LoRA for partial fine-tuning and no LoRA for full fine-tuning; `repos/moshi-finetune/finetune/wrapped_model.py:173-184` freezes or unfreezes parameters accordingly.
- **Text padding downweighting**: `repos/moshi-finetune/train.py:260-265` and `repos/moshi-finetune/finetune/loss.py:19-22` implement `text_padding_weight`; `repos/moshi-finetune/example/moshi_7B.yaml:18-19` sets `.5`.
- **Semantic codebook emphasis**: `repos/moshi-finetune/train.py:267-273` and `repos/moshi-finetune/finetune/loss.py:16-18` implement `first_codebook_weight_multiplier`; `repos/moshi-finetune/example/moshi_7B.yaml:18` sets `100.`.
- **Condition plumbing**: `repos/moshi-finetune/train.py:248-252` prepares condition tensors when `batch.condition_attributes` is present.
- **Checkpoint export**: `repos/moshi-finetune/finetune/checkpointing.py:199-246` writes safetensors checkpoints; `repos/moshi-finetune/finetune/checkpointing.py:54-59` distinguishes `lora.safetensors` from `consolidated.safetensors`.

These settings mean the repo is closest to the paper's **speech translation fine-tuning stage**, not to text pretraining, audio pretraining, or full speech translation training from scratch.

## Current Data Contract

The current repository expects a Moshi conversation-style dataset:

- JSONL file with rows like `{"path": "data_stereo/a.wav", "duration": 24.5}`.
- Each audio file has a matching sidecar `.json`.
- Sidecar JSON contains word alignments:

```json
{
  "alignments": [
    ["hello", [0.10, 0.40], "SPEAKER_MAIN"]
  ]
}
```

`InterleavedTokenizer.__call__`:

1. Receives one waveform chunk.
2. Encodes it with Mimi.
3. Loads sidecar alignments.
4. Builds one text stream aligned to the Mimi frame rate.
5. Concatenates:

```text
codes = [text_tokens, audio_tokens]
```

This is suitable for Moshi dialogue fine-tuning where the dataset is represented as a single multichannel audio file and one transcript stream. It is not enough for Hibiki S2ST.

## Hibiki Model Contract

Hibiki uses the same Moshi-family hierarchical LM structure, but with a translation-specific stream layout.

From the local Hibiki review and checkpoint analysis:

- Hibiki is decoder-only simultaneous S2ST/S2TT.
- It uses Mimi tokens at 12.5 Hz.
- It predicts target text as an Inner Monologue scaffold.
- It jointly models target speech, source speech, and target text.
- Source audio tokens are modeled during training but supplied from real source audio at inference.
- Speaker similarity is conditioned by a `description` LUT with values:
  - `very_bad`
  - `bad`
  - `neutral`
  - `good`
  - `very_good`

The local Hibiki checkpoints encode this layout directly.

### Hibiki-M / 1B

Config facts:

- `model_type: "hibiki"`
- `n_q: 16`
- `dep_q: 8`
- `delays` length: 17
- `text_card: 48000`
- `card: 2048`
- `conditioners.description` present

Stream interpretation:

```text
0       target text
1-8     generated target audio codebooks
9-16    source/input audio codebooks
```

Total training tensor shape:

```text
[B, 17, T]
```

### Hibiki / 2B

Config facts:

- `model_type: "hibiki"`
- `n_q: 32`
- `dep_q: 16`
- `delays` length: 33
- `text_card: 48000`
- `card: 2048`
- `conditioners.description` present
- `depformer_weights_per_step_schedule` shares later codebook-step weights onto index 8

Stream interpretation:

```text
0        target text
1-16     generated target audio codebooks
17-32    source/input audio codebooks
```

Total training tensor shape:

```text
[B, 33, T]
```

## Moshi Loader Compatibility

The local Moshi loader supports Hibiki checkpoints.

Evidence:

- `CheckpointInfo.from_hf_repo(...)` reads `config.json`.
- It pops `model_type` from the config, defaulting to `"moshi"`.
- It resolves `moshi_name`, `mimi_name`, and `tokenizer_name` from the config.
- `get_moshi(...)` calls `get_moshi_lm(...)` with the parsed LM config.
- If `model_type == "hibiki"`, it patches the EOS embedding to reduce premature EOS behavior.

This means the first model-loading adaptation can be configuration-only:

```yaml
moshi_paths:
  hf_repo_id: "kyutai/hibiki-1b-pytorch-bf16"
```

or local-path based:

```yaml
moshi_paths:
  hf_repo_id: null
  config_path: "/path/to/config.json"
  moshi_path: "/path/to/hibikim-pytorch-37c6cfd6@200.safetensors"
  mimi_path: "/path/to/mimi-pytorch-e351c8d8@125.safetensors"
  tokenizer_path: "/path/to/tokenizer_spm_48k_multi6_2.model"
```

The expected harder work is not loading the checkpoint. It is producing correct Hibiki `codes`.

## Main Adaptation Boundary

The main boundary is `finetune/data/interleaver.py`.

Current output:

```text
[target_or_main_text, one encoded audio group]
```

Required Hibiki output:

```text
[target_text, target_audio_codebooks, source_audio_codebooks]
```

A Hibiki-capable tokenizer should:

1. Load source audio.
2. Load target audio.
3. Encode source audio with frozen Mimi.
4. Encode target audio with frozen Mimi.
5. Build target text tokens at Mimi frame rate from aligned target-language timestamps.
6. Select the first `dep_q` target-audio codebooks.
7. Select the first `n_q - dep_q` source-audio codebooks.
8. Concatenate streams in Hibiki order.
9. Pad missing frames with `zero_token_id`.
10. Return optional speaker-similarity condition attributes.

The minimal target structure is:

```python
codes = torch.cat(
    [
        target_text_tokens,       # [1, 1, T]
        target_audio_tokens,      # [1, dep_q, T]
        source_audio_tokens,      # [1, n_q - dep_q, T]
    ],
    dim=1,
)
```

## Proposed Hibiki Dataset Schema

A minimal JSONL row for Hibiki fine-tuning should not reuse the current single-`path` shape. It should explicitly name both sides of the translation pair.

Recommended row:

```json
{
  "source_audio": "fr/source_000001.wav",
  "target_audio": "en/target_000001.wav",
  "duration": 52.3,
  "target_alignment": "alignments/target_000001.json",
  "speaker_similarity": "good",
  "source_language": "fr",
  "target_language": "en",
  "source_eos_frame": 651
}
```

Recommended target alignment sidecar:

```json
{
  "alignments": [
    ["the", [0.32, 0.45], "SPEAKER_MAIN"],
    ["food", [0.45, 0.71], "SPEAKER_MAIN"]
  ]
}
```

Optional fields for later paper-style reproduction:

- `source_transcript`
- `target_text`
- `alignment_method`
- `contextual_alignment_version`
- `tts_model`
- `tts_generation_rank`
- `speaker_similarity_score`
- `split`
- `dataset_source`
- `license`

## Condition Handling

Hibiki checkpoints include a `description` condition table. The training data should provide a condition per sample:

```python
ConditionAttributes(
    text={"description": "good"},
    tensor={},
)
```

For classifier-free-guidance-compatible fine-tuning, the condition labels should use only the values the checkpoint config supports:

```text
very_bad
bad
neutral
good
very_good
```

If speaker similarity is not computed yet, use a conservative fixed label such as `neutral` for training experiments, but treat that as a baseline simplification rather than paper-faithful Hibiki reproduction.

## Loss Implications

The current train loop computes loss over:

```text
text stream: codes[:, :model.audio_offset]
audio streams: codes[:, model.audio_offset:model.audio_offset + model.dep_q]
```

For Hibiki this means:

- text loss applies to target text
- audio loss applies to generated target audio
- source/input audio streams are included in the conditioning context but not directly included in this audio loss slice

This is compatible with a practical first Hibiki LoRA adaptation, because source audio is the input condition at inference. However, it is not guaranteed to reproduce paper training exactly.

The Hibiki review records that paper training and fine-tuning apply loss to both source and target streams. The current `LMModel.forward()` returns audio logits for `dep_q`, not for all `n_q` audio streams. Therefore, source-stream prediction loss would require deeper changes than just writing a new dataset tokenizer.

## What Can Be Reused Directly

Reusable with little or no change:

- `TrainArgs` config structure
- `ModelPaths`
- LoRA settings
- AdamW optimizer settings
- mixed precision helpers
- FSDP wrapping policy
- checkpoint writing
- TensorBoard/W&B logging
- text/audio masked CE helper
- data-source weighting and sharding concept

Reusable with focused changes:

- `Interleaver.build_token_stream(...)` for target text alignment
- JSONL data-source parsing
- batch collation
- condition-attribute plumbing
- evaluation loss loop

Not reusable as-is:

- single-audio-file dataset assumption
- Moshi conversation sidecar annotation format as the only input format
- inference instructions using `moshi.server`
- metric interpretation as generic LM perplexity only

## Practical Implementation Phases

### Phase 1: Loader Smoke Test

Goal: prove `moshi-finetune` can instantiate Hibiki under its current training stack.

Tasks:

1. Create a local Hibiki YAML based on `example/moshi_7B.yaml`.
2. Point `moshi_paths` to `kyutai/hibiki-1b-pytorch-bf16` or local checkpoint files.
3. Disable checkpointing and use a tiny synthetic batch.
4. Confirm model loads, LoRA initializes, and a forward pass accepts `[B, 17, T]`.

Stop condition:

- One forward pass returns text logits and audio logits without shape errors.

### Phase 2: Hibiki Tokenizer

Goal: replace single-audio tokenization with paired source-target S2ST token packing.

Tasks:

1. Add a new dataset schema for source audio, target audio, and target alignment.
2. Add a Hibiki tokenizer/interleaver that emits `[target_text, target_audio, source_audio]`.
3. Return `ConditionAttributes` for speaker similarity.
4. Add shape checks for 1B and 2B:
   - 1B: `codes.shape[1] == 17`
   - 2B: `codes.shape[1] == 33`

Stop condition:

- A real sample from the planned dataset produces valid `codes` for the selected Hibiki checkpoint.

### Phase 3: Minimal LoRA Fine-Tune

Goal: train a small adapter using target text and target audio loss.

Tasks:

1. Use a small open dataset slice, likely CVSS/CoVoST-derived.
2. Use `duration_sec` below the paper maximum for a memory-safe first run.
3. Keep Mimi frozen.
4. Train LoRA only.
5. Export `lora.safetensors`.

Stop condition:

- Training loss decreases on a small validation slice.
- Inference with the adapter still runs through `moshi.run_inference` or an equivalent Hibiki path.

### Phase 4: Paper-Style Gap Closure

Goal: move from practical adaptation toward Hibiki-like training.

Tasks:

1. Implement or approximate contextual alignment.
2. Generate or select target speech with causal alignment.
3. Compute speaker-similarity labels.
4. Add source-audio augmentation.
5. Investigate whether source-stream loss can be supported through a modified model forward path.

Stop condition:

- The training recipe documents each divergence from the paper and has objective metrics for translation quality, speaker similarity, and latency.

## Open Questions

These are the main ambiguity points before implementation:

1. Should the first implementation target Hibiki-M 1B for on-device relevance, or Hibiki 2B for fidelity?
2. Should the first training pass use only target text plus target audio loss, or should it attempt paper-style source-stream loss from the beginning?
3. Should the dataset recipe start from CVSS/CoVoST because it is reproducible, or from a custom synthetic pipeline closer to Hibiki's 40K-hour construction?
4. Should speaker-similarity conditioning begin with fixed labels, heuristic labels, or a verifier-based scoring pipeline?
5. Is the near-term goal a runnable fine-tuning proof of concept, or a research-faithful Hibiki reproduction?

## Recommended Next Step

Use a narrow proof first:

1. Select `kyutai/hibiki-1b-pytorch-bf16`.
2. Build a tiny two-audio-file Hibiki dataset fixture outside Git.
3. Implement only the new tokenizer/interleaver path.
4. Run one forward/backward step with `do_ckpt: false`.
5. Then decide whether to invest in source-stream loss and contextual-alignment data generation.

This sequence tests the highest-risk code boundary, the S2ST stream packing, before spending effort on large data preparation.
