# lfm-mlx — merge, convert & quantize LFM2.5-2.6B for Apple Silicon

Taking my QLoRA tool-calling adapter and turning it into a model that actually **runs on-device**: **merge** the LoRA adapter back into the base weights, **convert** to Apple's **MLX** format, **quantize** it to **8-bit**, and publish all three artifacts to the Hugging Face Hub.

> A **learning-in-public** project — the deployment step after [fine-tuning LFM2.5-2.6B with QLoRA](https://github.com/Hemeskyo/lfm-qlora) and [building a transformer from scratch](https://github.com/Hemeskyo/miniGPT).

---

## Why this step exists

The QLoRA project produced a **LoRA adapter** — a few hundred MB of bf16 low-rank updates that mean nothing on their own; they only have meaning *applied on top of the base model*. (The 4-bit quantization in QLoRA was a **training-time** trick to fit the frozen base in VRAM — the adapter matrices themselves are bf16, and they can be merged back onto the **full-precision 16-bit** base, which is exactly what happens here.) That's great for training, but a loose adapter isn't a shippable on-device model. To actually run this on a Mac (or, eventually, an iPhone), two things have to happen:

1. **Merge** — fold the adapter's `W + (α/r)·B·A` update permanently back into the full-precision base, producing a single standalone model with no PEFT dependency.
2. **Convert + quantize** — re-express that model in **MLX**, Apple's array framework built for the unified-memory architecture of Apple Silicon, and shrink it with **8-bit quantization** so it fits comfortably in RAM and decodes fast.

| Framework | Built for | Why it matters here |
|---|---|---|
| **PyTorch + PEFT** | Training (CUDA) | Where the adapter was born; used here only to **merge** |
| **MLX** | Inference on Apple Silicon | Native unified-memory arrays — the base and KV-cache live in the **same memory** the GPU reads, no host↔device copies |

---

## The pipeline

```
Hskyto/toolcall_adapter  (LoRA, from the QLoRA project)
            │
            │  merge.py  —  PEFT merge_and_unload() into bf16 base
            ▼
     merged_model/                      (~5.1 GB, standalone PyTorch bf16)
            │
            │  mlx_lm.convert            (no quantization)
            ▼
 lfm2.5-2.6b-toolcall-mlx/              (~5.1 GB, MLX 16-bit)
            │
            │  mlx_lm.convert  -q  --q-bits 8   (group_size 64, affine)
            ▼
 lfm2.5-2.6b-toolcall-mlx-q8/           (~2.7 GB, MLX 8-bit)  ←  the on-device model
            │
            │  huggingface-cli upload
            ▼
         Hugging Face Hub
```

### 1 · Merge — `merge.py`

Loads `LiquidAI/LFM2.5-2.6B-Base` in **bfloat16**, applies the adapter with `PeftModel.from_pretrained`, then `merge_and_unload()` bakes the low-rank update into the base weights and drops the PEFT wrappers. The result is saved as a plain `transformers` model — architecturally identical to the base, but now carrying the tool-calling behaviour. Device auto-selects **CUDA → MPS → CPU**, and the tokenizer is copied from the adapter (falling back to the base) so the chat template travels with the weights.

### 2 · Convert to MLX — 16-bit

```bash
mlx_lm.convert --hf-path merged_model -q False --mlx-path lfm2.5-2.6b-toolcall-mlx
```

A faithful, full-precision MLX copy of the merged model — the reference point to check that conversion itself didn't change behaviour before quantizing.

### 3 · Quantize to MLX — 8-bit

```bash
mlx_lm.convert --hf-path merged_model -q --q-bits 8 --mlx-path lfm2.5-2.6b-toolcall-mlx-q8
```

**Affine 8-bit quantization, `group_size = 64`** — weights are stored as int8 with a scale/zero-point shared across each group of 64. This roughly **halves the footprint (5.1 GB → 2.7 GB)** with negligible quality loss at 8-bit, which is the sweet spot for a 2.6B model on a Mac.

| Artifact | Format | Size | Role |
|---|---|---|---|
| `merged_model/` | PyTorch bf16 | ~5.1 GB | Standalone merged base (conversion source) |
| `lfm2.5-2.6b-toolcall-mlx/` | MLX 16-bit | ~5.1 GB | Full-precision MLX reference |
| `lfm2.5-2.6b-toolcall-mlx-q8/` | MLX 8-bit | ~2.7 GB | **The on-device deliverable** |

*(Model: LFM2.5-2.6B — hybrid conv+attention, 30 layers, hidden 2048, vocab 128k.)*

### 4 · Publish

All three are pushed to the Hub:
- 🤗 [`Hskyto/toolcall_adapter`](https://huggingface.co/Hskyto/toolcall_adapter) — the source LoRA adapter
- 🤗 [`Hskyto/lfm2.5-2.6b-toolcall-mlx`](https://huggingface.co/Hskyto/lfm2.5-2.6b-toolcall-mlx) — MLX 16-bit
- 🤗 [`Hskyto/lfm2.5-2.6b-toolcall-mlx-q8`](https://huggingface.co/Hskyto/lfm2.5-2.6b-toolcall-mlx-q8) — MLX 8-bit

---

## Verifying it runs — `test.py`

The whole point is inference on Apple Silicon, so `test.py` loads the **8-bit MLX** model with `mlx_lm` and runs **12 real-world on-device queries** ("Remind me to call Mom at 6pm", "Turn off Wi-Fi and enable low power mode", "Set an alarm for 06:00 called Early Flight"…) against a **6-tool iOS registry** (`create_reminder`, `schedule_event`, `send_message`, `set_alarm`, `adjust_system_setting`, `start_timer`).

Each query is rendered through the model's **own chat template** with the tools injected (`tokenizer.apply_chat_template(messages, tools=IOS_TOOLS, ...)`) — the same format the model was fine-tuned on — and the model emits its Pythonic tool call:

```
Query: "Remind me to call Mom at 6pm"
-> <|tool_call_start|>[ios.create_reminder(text='call Mom', time='18:00')]<|tool_call_end|>
```

Loading and generating happen entirely on-device, in unified memory, with no GPU/CUDA anywhere in sight — which was the goal.

---

## Project structure

```
merge.py                        # Merge the LoRA adapter into the bf16 base -> merged_model/
test.py                         # Load the 8-bit MLX model with mlx_lm, run 12 on-device iOS cases
merged_model/                   # Standalone merged model (PyTorch bf16)  [gitignored]
lfm2.5-2.6b-toolcall-mlx/       # MLX 16-bit conversion                    [gitignored]
lfm2.5-2.6b-toolcall-mlx-q8/    # MLX 8-bit quantization (the deliverable) [gitignored]
toolcall_adapter/               # The source LoRA adapter                  [gitignored]
LFM_2.6B_BASE_MODEL/            # Local copy of the base model             [gitignored]
```

*(Weights are gitignored — they live on the Hub, not in the repo.)*

---

## How to run

```bash
# 1. Setup (Apple Silicon)
python -m venv .venv && source .venv/bin/activate
pip install mlx-lm transformers peft torch huggingface_hub

# 2. Merge the adapter into the base
python merge.py                 # -> merged_model/

# 3. Convert to MLX, then quantize to 8-bit
mlx_lm.convert --hf-path merged_model -q False   --mlx-path lfm2.5-2.6b-toolcall-mlx
mlx_lm.convert --hf-path merged_model -q --q-bits 8 --mlx-path lfm2.5-2.6b-toolcall-mlx-q8

# 4. Run the on-device tool-calling test
python test.py                  # loads the q8 MLX model, runs the 12 iOS cases

# 5. (optional) publish
huggingface-cli upload Hskyto/lfm2.5-2.6b-toolcall-mlx-q8 lfm2.5-2.6b-toolcall-mlx-q8 .
```

You can point `test.py` at the 16-bit build instead by switching `model_path` to `Hskyto/lfm2.5-2.6b-toolcall-mlx`.

---

## What I learned

- **Adapter → shippable model**: why a LoRA adapter isn't deployable on its own, and how `merge_and_unload()` folds the low-rank update permanently into the base.
- **MLX and unified memory**: why Apple Silicon inference wants a native array framework — weights and KV-cache in the same memory the GPU reads, no host↔device copies — instead of PyTorch/CUDA.
- **Quantization in practice**: affine 8-bit with `group_size 64` roughly halves the footprint (5.1 GB → 2.7 GB) for a 2.6B model with negligible quality loss — and why 8-bit is the comfortable default over 4-bit at this size.
- **The chat template is the contract**: inference has to render tools the exact way training did, or the fine-tune doesn't transfer — the same lesson from the QLoRA project, now on the inference side.
- **The full deployment loop end-to-end**: adapter → merge → MLX convert → quantize → Hub → local Apple Silicon inference. The training half was the previous project; this is the on-device half.
