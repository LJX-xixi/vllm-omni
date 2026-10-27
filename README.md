# Qwen3-Omni chunked-prefill text-loss reproducer

This companion to the PR contains a fixed 16-prompt input set, an offline
inference driver, a deployment YAML, a metadata observer and an ASR scorer.
It does not require the author's research repository, machine directories,
Python environment or GPU scheduler.

## Environment

Start in a clean checkout of the PR branch. For the recorded Linux x86_64
CUDA setup, use Python 3.12 and a CUDA 13-compatible driver. The repository's
[CUDA source-install instructions](https://github.com/vllm-project/vllm-omni/blob/63e55950a6ec7f0b82020ac2978a804faf673143/docs/getting_started/installation/gpu/cuda.inc.md)
pin the vLLM wheel used in the experiment. A fresh-environment recipe is:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --torch-backend=cu130 \
  'vllm @ https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'torch==2.13.0' 'transformers==5.16.1' -e '.[dev]'
```

The repository dev dependencies include pytest, openai-whisper 20250625
and jiwer 4.0.0. Install the `ffmpeg` executable for ASR. This recipe follows
the pinned checkout's installation guide; it has not been independently
tested by rebuilding an empty environment for this PR. The recorded run
used those vLLM/PyTorch/Transformers versions in an existing environment.

The CPU regression requires no model weights or GPU. From the PR checkout:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest \
  tests/model_executor/models/qwen3_omni/test_qwen3_omni_mrv2.py \
  -m 'core_model and cpu' --run-level=core_model -q
```

## GPU comparison

Use two CUDA GPUs with enough memory for this model and deployment. The
recorded pair used two H200-class/sm90 devices; other hardware has not been
validated here. Obtain GPU access through your own environment's mechanism.
The YAML's device IDs 0 and 1 refer to visible devices, not physical card IDs.

Put the ZIP next to the Omni checkout, then run from the checkout root:

```bash
unzip ../qwen3-prefill-reproducer.zip -d ..
HARNESS="$PWD/../qwen3-prefill-reproducer"
OMNI_SRC="$PWD"
MODEL="$HARNESS/model"
RUN_ROOT="$HARNESS/results"
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --revision 26291f793822fb6be9555850f06dfe95f2d7e695 \
  --local-dir "$MODEL"
```

An existing local copy of that model revision may be used by changing
`MODEL`; no specific cache layout is required. Choose a fresh `RUN_ROOT`
for each comparison. Model acquisition requires network access unless the
snapshot is already available; inference below uses the local snapshot.

Start with the PR checked out. Within your two-GPU allocation, run the arms
sequentially in the same checkout and environment. Each driver closes its
engine before returning. The following commands do not allocate GPUs:

```bash
PR_COMMIT=$(git rev-parse HEAD)
PR_BRANCH=$(git branch --show-current)
PAIR_RC=0
for ARM in before after; do
  if [ "$ARM" = before ]; then
    git switch --detach 566d9521759efd40940ad9c567f1269da3475286 || { PAIR_RC=$?; break; }
  else
    git switch --detach "$PR_COMMIT" || { PAIR_RC=$?; break; }
  fi
  mkdir -p "$RUN_ROOT/$ARM/stress_chunk_async/hook"
  git rev-parse HEAD
  PYTHONPATH="$HARNESS/hook:$OMNI_SRC" \
  Q3TS_HOOK_DIR="$RUN_ROOT/$ARM/stress_chunk_async/hook" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python "$HARNESS/drive.py" \
    --model "$MODEL" \
    --deploy-config "$HARNESS/deploy/stress_chunk_async.yaml" \
    --prompts "$HARNESS/prompts.txt" \
    --thinker-max-tokens 120 --talker-max-tokens 640 \
    --out "$RUN_ROOT/$ARM/stress_chunk_async" || { PAIR_RC=$?; break; }
done
if [ -n "$PR_BRANCH" ]; then git switch "$PR_BRANCH"; fi
test "$PAIR_RC" -eq 0
```

Each arm must complete 16/16 requests with `stop` finishes, identical
Thinker text between arms, observed nonzero-offset prefill, and no replay.
If these conditions do not hold, the comparison does not isolate this bug.
The observer records metadata without changing tensors, but adds
synchronization and file I/O. Do not interpret elapsed time as performance.

## CPU scoring

The scorer uses Whisper's standard cache by default, downloading `base`
on its first run if needed. Use `--asr-cache DIRECTORY` to select another
cache. No author's cache or interpreter path is required:

```bash
CUDA_VISIBLE_DEVICES='' python "$HARNESS/score.py" \
  "$RUN_ROOT/before" --decode-row counter
CUDA_VISIBLE_DEVICES='' python "$HARNESS/score.py" \
  "$RUN_ROOT/after" --decode-row position
```

The output includes generated WAVs, Thinker text, exact request IDs, trace
records, ASR transcripts, per-request WER and second-word deletion flags.
WAVs are model-generated outputs, not ground-truth recordings. Cache-read
counts reconstruct the index used by each source revision; CPU regression
tests independently verify exact tensor-row order. WER is the unweighted
mean across requests. Different devices or schedules need not reproduce
identical waveforms or WER values.

## Relationship to the recorded run

The recorded GPU pair used baseline `566d9521759efd40940ad9c567f1269da3475286`
and candidate `b37f2a5216819f1e80e53ada8bec7fe59584b8be`. The current PR model
file is identical to that candidate. These instructions reproduce the
procedure using the PR head, not the exact historical source tree.

The shared driver, prompts and scoring algorithm retain their recorded
behavior. For portability, the observer no longer loads a host-specific
Python `sitecustomize.py`, and the scorer's cache is optional rather than
an absolute directory. YAML settings are unchanged; its header now describes
the actual comparison. CPU checks cover the adjusted observer/scorer.
GPU inference has not been repeated with these packaging changes, and
cross-machine execution is not claimed as independently validated.
