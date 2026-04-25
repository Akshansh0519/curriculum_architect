# 🐛 MISTAKES.md — THE EXAMINER
## Living AI Session Error Log · Team Institutional Memory
**Project:** The Examiner · **Hackathon:** OpenEnv India 2026 · **Submission Deadline:** TBD

---

# Pre-Session Ritual Prompt (Paste into AI at Session Start)

```
Project: The Examiner — an RL environment training diagnostic question design.
Stack: OpenEnv (openenv-core), Unsloth + TRL GRPOTrainer, Qwen2.5-7B, W&B, Gradio on HF Space.
Constraints:
- DO NOT write custom training loops (use GRPOTrainer)
- DO NOT reimplement OpenEnv base classes (inherit from Environment)
- DO NOT use LLM-as-judge in reward function (fully programmatic)
- DO NOT add observation/action space fields beyond spec
- DO NOT use local file paths (Colab environment)
- DO NOT commit video files to HF Hub
- All environment methods must be async
- Reward formula: R = 0.70*acc - 0.50*FA + 0.20*eff + 0.10*diag
Current phase: [UPDATE THIS]
Open MSRs: [UPDATE THIS]
Read guardrails.md for full constraints.
```

---

# Recurring Mistakes Index (Keep at Top — Always Current)

| # | Category | Description | Session | Severity | MSR Risk |
|---|---|---|---|---|---|
| | | *No mistakes logged yet* | | | |

---

# RL/OpenEnv Pre-Populated Mistake Patterns

> These are mistakes AI tools commonly make in projects like ours. Pre-loaded so the team can recognize them instantly.

| # | Category | Pattern | MSR Risk | Detection | Prevention |
|---|---|---|---|---|---|
| PP-1 | OPENENV_MISUSE | AI reimplements OpenEnv `Environment` base class instead of inheriting | MSR-1 🔴 | Check: does class have `(Environment)` in declaration? | Always paste base class interface into AI context |
| PP-2 | TRAINING_FAILURE | AI generates custom training loop instead of using `GRPOTrainer` | MSR-2 🔴 | Search for `GRPOTrainer` in training code — must be present | Prompt: "Use TRL GRPOTrainer. Do not write a custom training loop." |
| PP-3 | REWARD_LOGIC_ERROR | AI generates fake/mocked training plots with synthetic data | MSR-3 🔴 | Check matplotlib code: does it pull from W&B or generate fake data? | "Pull data from wandb.Api(). Do not generate synthetic plot data." |
| PP-4 | WRONG_VERSION | AI uses deprecated `push_to_hub()` API pattern | MSR-5 🟡 | Runtime error on push | Pin `huggingface_hub>=0.27`, verify API call against current docs |
| PP-5 | HF_DEPLOY_FAIL | AI generates HF Space config that looks valid but fails at runtime | MSR-5 🟡 | Space loads but Gradio doesn't render | Always test in incognito browser after deployment |
| PP-6 | REWARD_LOGIC_ERROR | AI writes reward function that always returns same value | MSR-3 🔴 | Test with different inputs — reward doesn't change | Unit test: 6 different scenarios must produce 6 different rewards |
| PP-7 | TRAINING_FAILURE | AI generates Colab cells with local file path assumptions | MSR-2 🟡 | Cell fails with FileNotFoundError | Search for `/home/`, `C:\`, `../` in notebook |
| PP-8 | SCOPE_DRIFT | AI adds video files to HF Hub repo | MSR-9 🔴 | `git ls-files` shows media files | Never add .mp4/.avi/.mov/.webm — use external URLs |
| PP-9 | SCOPE_DRIFT | AI expands observation/action spaces beyond architecture.md spec | MSR-1 🟡 | Compare models.py fields against architecture.md | "Use exactly these fields, no more: [list]" |
| PP-10 | SILENT_FAILURE | AI skips W&B logging hooks in training code | MSR-3 🟡 | W&B dashboard empty after training run | Always verify `wandb.log()` calls in training loop |
| PP-11 | HALLUCINATION | AI generates sync `def step()` instead of `async def step()` | MSR-1 🟡 | OpenEnv client fails to connect | "All environment methods must be async def" |
| PP-12 | INTEGRATION_BREAK | AI uses `StepResult` as tuple `(obs, reward, done, info)` | MSR-1 🟡 | Type error at runtime | "Return StepResult(observation=..., reward=..., done=..., truncated=..., info=...)" |

---

# Per-Session Log Template

```markdown
## Session [number] — [date] — [coder: C1/C2/VAL]

### Work Done
- [list tasks completed]

### Mistakes Encountered

#### Mistake [session]-[number]
- **Category:** [HALLUCINATION | WRONG_VERSION | SCOPE_DRIFT | INTEGRATION_BREAK | GIT_CONFLICT | MSR_VIOLATION | OPENENV_MISUSE | TRAINING_FAILURE | HF_DEPLOY_FAIL | REWARD_LOGIC_ERROR | SILENT_FAILURE | OWNERSHIP_VIOLATION]
- **Severity:** 🔴/🟡/🟢
- **File:** [path]
- **Description:** [what went wrong]
- **MSR at risk:** [MSR-N or NONE]
- **Judging criterion impacted:** [ENV_INNOV/STORYTELLING/REWARD_EVIDENCE/PIPELINE or NONE]
- **Original prompt that caused it:** [paste the prompt]
- **Root cause:** [why the AI generated this]
- **Fix:** [what was changed]
- **Corrected prompt:** [the prompt that produced the correct output]
- **Time lost:** [minutes]

### MSR Status After Session
- MSR-1: [OPEN/CLOSED]
- MSR-2: [OPEN/CLOSED]
- MSR-3: [OPEN/CLOSED]
- MSR-4: [OPEN/CLOSED]
- MSR-5: [OPEN/CLOSED]
- MSR-6: [OPEN/CLOSED]
- MSR-7: [OPEN/CLOSED]
- MSR-8: [OPEN/CLOSED]
- MSR-9: [OPEN/CLOSED]
```

---

# Session Logs

*No sessions logged yet.*
