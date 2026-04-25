# 🔧 CODER 2 — Implementation Playbook
## The Examiner · Training Pipeline & Deployment Specialist

---

# Role Summary
C2 owns the **training pipeline, deployment, and presentation infrastructure**: GRPO training script, Colab notebook, W&B integration, HF Space, plot generation, and all deployment artifacts. C2's work directly drives **Reward Evidence (20%)**, **Pipeline (10%)**, and supports **Storytelling (30%)**.

# File Ownership (Absolute — No Exceptions)
```
training/config.py                ← C2
training/train_grpo.py            ← C2
training/train_colab.ipynb        ← C2
training/eval.py                  ← C2
scripts/generate_plots.py         ← C2
scripts/select_transcripts.py     ← C2
scripts/push_to_hub.py            ← C2
hf_space/app.py                   ← C2
hf_space/requirements.txt         ← C2
hf_space/README.md                ← C2
hf_space/assets/*                 ← C2
examiner_env/client.py            ← C2
outputs/plots/*                   ← C2
outputs/transcripts/*             ← C2
```

# MSR Ownership
| MSR | C2 Responsibility |
|---|---|
| MSR-2 | **PRIMARY** — Training script + Colab notebook |
| MSR-3 | **PRIMARY** — Real training plots from W&B |
| MSR-5 | **PRIMARY** — HF Space live and runnable |
| MSR-9 | **PRIMARY** — No video files in HF Hub |

# Judging Criterion Ownership
| Criterion | C2 Role |
|---|---|
| REWARD_EVIDENCE (20%) | **PRIMARY** — plots, curves, before/after evidence |
| PIPELINE (10%) | SHARED — training script execution |
| STORYTELLING (30%) | SUPPORT — HF Space, plots, transcript artifacts |

# Parallel Work Map
| When C1 Does... | C2 Does... | Sync Point |
|---|---|---|
| S0.1 (scaffold) + S0.3 (KB) | S0.2 (Colab skeleton + W&B) | Phase 0 gate |
| S1.1-S1.4 (env core) | Prepare training script skeleton, config | Phase 1 gate |
| Reward tests, student tuning | S2.1-S2.4 (training pipeline) | Phase 2 gate |
| S3.4 (adaptive difficulty) | S3.1 (training run) + S3.2 (plots) + S3.3 (transcripts) | Phase 3 gate |
| Support | S4.1 (HF Space) + S4.2 (README support) | Phase 4 gate |

---

# Tasks

## S0.2 — Colab Notebook Skeleton + W&B Setup
**Goal:** Create Colab notebook skeleton with all installs and imports  
**Criterion:** PIPELINE | **MSR:** MSR-2 (partial) | **Time:** 25 min

**AI Prompt:**
```
Create a Google Colab notebook (training/train_colab.ipynb). Cells:

Cell 1 - Installs:
  !pip install unsloth[colab] trl>=0.15 wandb>=0.19 openenv-core gradio>=5.0 huggingface_hub>=0.27 transformers accelerate peft

Cell 2 - Imports:
  from unsloth import FastLanguageModel
  from trl import GRPOTrainer, GRPOConfig
  import wandb
  import torch, os

Cell 3 - W&B Init:
  wandb.login()
  wandb.init(project='the-examiner', config={...})

Cell 4 - Model Load:
  model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=4096,
    load_in_4bit=True,
  )
  model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.05)

NO local file paths. Everything installs from pip or HF Hub. Each cell has a markdown header explaining what it does.
```

**Sanity check:**
1. Zero local file path references (no `/home/user/`, no `C:\`, no relative `../`)
2. All imports resolve after Cell 1 pip installs
3. W&B init cell has project name 'the-examiner'

---

## S2.1 — Training Script (GRPO)
**Goal:** Full GRPO training script using Unsloth + TRL  
**Criterion:** PIPELINE | **MSR:** MSR-2 | **Time:** 60 min

**AI Prompt:**
```
Create training/train_grpo.py. Use Unsloth FastLanguageModel + TRL GRPOTrainer.

1. Load Qwen2.5-7B-Instruct via Unsloth with LoRA (r=16, alpha=32, dropout=0.05, 4-bit)
2. Define TrainingConfig dataclass (all hyperparameters, no hardcoding):
   - learning_rate=5e-6, max_steps=500, per_device_batch_size=1
   - num_generations=4, kl_penalty=0.01
   - wandb_project='the-examiner'
   - eval_every=50, save_every=100

3. Define reward_fn(completions, prompts) -> list[float]:
   - For each completion, parse as ExaminerAction sequence
   - Run against ExaminerEnvironment episode
   - Return list of episode rewards

4. Configure GRPOConfig with all params from TrainingConfig
5. Init GRPOTrainer(model=model, config=config, tokenizer=tokenizer, reward_funcs=[reward_fn])
6. trainer.train()
7. After training: model.save_pretrained_merged("examiner_checkpoint")
8. Push to HF Hub

Log to W&B per episode: total_reward, accuracy, false_accusations, efficiency_score, mean_turns_to_classify, loss, kl_divergence.

CRITICAL: Use GRPOTrainer. Do NOT write a custom training loop. This is MSR-2.
```

**Context files:** `architecture.md` (Section 5), `examiner_env/` module  
**Sanity check:**
1. Uses `GRPOTrainer` — search for "GRPOTrainer" in file
2. `wandb.log()` calls present for all 7 metrics
3. No hardcoded hyperparameters — all from config

---

## S2.2 — Colab Notebook Integration
**Goal:** Merge training script into Colab cells  
**Criterion:** PIPELINE | **MSR:** MSR-2 | **Time:** 30 min

**AI Prompt:**
```
Update training/train_colab.ipynb. Convert train_grpo.py into Colab cells:

Cell 5 - TrainingConfig:
  @dataclass with all hyperparameters

Cell 6 - Environment Setup:
  from examiner_env import ExaminerEnvironment, ExaminerAction, ExaminerObservation
  (install examiner_env from HF Space or GitHub in Cell 1)

Cell 7 - Reward Function:
  def reward_fn(completions, prompts): ...

Cell 8 - GRPOTrainer Init:
  config = GRPOConfig(...)
  trainer = GRPOTrainer(...)

Cell 9 - Train:
  trainer.train()

Cell 10 - Save + Push:
  model.save_pretrained_merged(...)
  model.push_to_hub(...)

Cell 11 - Generate Plots:
  (plot generation code inline)

EVERY cell must assume Colab environment. No local paths. Cell 1 must install examiner_env package.
```

**Sanity check:**
1. Cell 1 installs everything including `examiner_env`
2. No cell references local filesystem
3. Cells run sequentially top-to-bottom

---

## S2.4 — Pipeline Smoke Test
**Goal:** Verify pipeline produces non-trivial improvement  
**Criterion:** PIPELINE | **MSR:** MSR-2, MSR-3 | **Time:** 30 min

**AI Prompt:**
```
Add smoke test cells to train_colab.ipynb:

Cell - Pre-Training Baseline:
  Run 20 episodes with untrained model. Log mean reward to wandb.
  baseline_reward = mean(rewards)
  print(f"Baseline mean reward: {baseline_reward}")

Cell - Quick Training:
  Run GRPOTrainer for 10 steps only (quick validation).

Cell - Post-Training Check:
  Run 20 episodes with trained model. Log mean reward.
  trained_reward = mean(rewards)
  print(f"Trained mean reward: {trained_reward}")
  assert trained_reward > baseline_reward, "Pipeline broken: no improvement after training"
```

**Sanity check:**
1. Baseline reward is logged
2. Trained reward is logged
3. Assertion passes (trained > baseline)

---

## S3.1 — Full Training Run
**Goal:** Run 500 episodes, full logging to W&B  
**Criterion:** REWARD_EVIDENCE | **MSR:** MSR-3 | **Time:** 4-6 hrs (GPU)

**AI Prompt:** Not applicable — this is execution, not code generation. Run the Colab notebook Cell 9 with full `max_steps=500`.

**Sanity check:**
1. W&B shows 500 episodes logged
2. Reward curve shows 3 phases (chance → learning → plateau)
3. No NaN values in loss or reward

---

## S3.2 — Plot Generation
**Goal:** Export publication-quality plots from W&B  
**Criterion:** REWARD_EVIDENCE | **MSR:** MSR-3 | **Time:** 30 min

**AI Prompt:**
```
Create scripts/generate_plots.py using wandb API + matplotlib.

1. Connect to wandb API, pull run history for project='the-examiner'
2. Plot 1: reward_curve.png
   - X: episode number, Y: mean reward (smoothed with rolling window=20)
   - Mark 3 phases with vertical dashed lines and labels: "Random" (0-100), "Learning" (100-300), "Skilled" (300-500)
   - Clean style: white background, labeled axes, legend

3. Plot 2: accuracy_curve.png
   - X: episode, Y: classification accuracy
   - Same phase markers

4. Plot 3: question_types.png
   - Side-by-side bar chart: question type distribution at episode 10 vs episode 400
   - Categories: Definitional, Procedural, Mechanistic, Causal, Edge-case

Save all to outputs/plots/. Use matplotlib style: seaborn-v0_8-whitegrid. DPI=150.
```

**Sanity check:**
1. 3 PNG files in `outputs/plots/`
2. Reward curve shows upward trend
3. Question type chart shows shift from definitional to mechanistic

---

## S4.1 — HuggingFace Space
**Goal:** Deploy Gradio app to HF Space  
**Criterion:** STORYTELLING | **MSR:** MSR-5 | **Time:** 60 min

**AI Prompt:**
```
Create hf_space/app.py using Gradio Blocks.

Tab 1 "🎯 Live Demo":
  - Dropdown: select difficulty (easy/medium/hard)
  - Button: "Run Examination Episode"
  - Chatbot component: shows Examiner questions and Student answers in real-time
  - Output: classification result vs ground truth, episode reward

Tab 2 "📊 Training Results":
  - Image: reward_curve.png (from assets/)
  - Image: accuracy_curve.png
  - Two textboxes side by side: "Before Training (Episode 10)" and "After Training (Episode 400)"
  - Load from assets/before.txt and assets/after.txt

Tab 3 "ℹ️ About":
  - Markdown: 3-sentence narrative (VERBATIM from guardrails.md)
  - Image: architecture diagram
  - Links: GitHub repo, blog post, training notebook

hf_space/README.md frontmatter:
---
title: The Examiner
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.0"
app_file: app.py
pinned: true
---

CRITICAL: Test in incognito browser after deployment.
```

**Sanity check:**
1. `sdk: gradio` in README frontmatter (not streamlit)
2. All 3 tabs render without errors
3. Live Demo tab can run an episode

---

# What C2 Must NOT Do
- ❌ Modify `examiner_env/models.py`, `reward.py`, `knowledge_base.py`, `student.py`
- ❌ Modify `examiner_env/server/examiner_environment.py`
- ❌ Change the reward function weights or formula
- ❌ Add observation/action space fields
- ❌ Commit to `main` branch directly

# AI Anti-Patterns for C2's Domain
| Anti-Pattern | Why Dangerous | Detection |
|---|---|---|
| Custom training loop | Violates MSR-2 | No `GRPOTrainer` in code |
| Local file paths in Colab | Notebook fails for judges | Search for `/home/`, `C:\`, `../` |
| Deprecated HF push API | Push fails silently | Check `huggingface_hub` version |
| `gr.Interface` instead of `gr.Blocks` | Less control for tabs | Check Gradio import |
| Skipping W&B logging | MSR-3 violation | No `wandb.log()` calls |
| Video files in HF repo | MSR-9 violation | Check `git ls-files` |
| Mocked/fake plots | MSR-3 violation — immediate 🔴 | Verify W&B timestamps match |

# Validator Handoff Template
```
HANDOFF C2 → VAL | Phase [N] Gate
Files changed: [list]
MSR status: MSR-2 [OPEN/CLOSED], MSR-3 [OPEN/CLOSED], MSR-5 [OPEN/CLOSED], MSR-9 [OPEN/CLOSED]
Sanity checks: [PASS/FAIL] for each
W&B run URL: [url]
HF Space URL: [url]
Known issues: [list or NONE]
Commit hash: [hash]
Branch: feat/c2-S[stage]-[feature]
```

# Local Environment Setup
```bash
git clone [repo-url]
cd the-examiner
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install openenv-core
pip install -e ./examiner_env
pip install unsloth trl wandb gradio huggingface_hub
pip install matplotlib pandas
pip install pytest
```

# Special C2 Instructions
1. **HF Space deployment:** AI tools generate Space configs that fail silently. After every deployment, open the Space URL in an **incognito browser**. If Gradio doesn't load → check `sdk_version` in README frontmatter and `requirements.txt` dependencies.
2. **Colab notebook:** The notebook must run **top-to-bottom without modification**. AI will generate cells that depend on local paths or previously-defined variables from a different session. After completing the notebook, do a **Kernel → Restart and Run All** test.
3. **Logging:** NEVER let AI skip W&B logging hooks. They are mandatory for MSR-3. If AI generates training code without `wandb.log()`, reject immediately.
4. **Plot authenticity:** Plots MUST come from real W&B data. If AI generates matplotlib code that creates synthetic data for "demonstration purposes," reject immediately. This is MSR-3 — fake plots are an instant 🔴 blocker.
