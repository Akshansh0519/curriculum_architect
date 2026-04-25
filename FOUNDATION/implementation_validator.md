# 🛡️ VALIDATOR — Implementation Playbook
## The Examiner · Quality Control & Submission Specialist

---

# Role Summary
The Validator ensures every merge is MSR-compliant, every output serves judging criteria, and the final submission is bulletproof. The Validator is also vibe coding — prompts are for reviewing, auditing, and verifying.

# File Ownership
```
tests/test_reward.py              ← VAL
tests/test_environment.py         ← VAL
tests/test_episode.py             ← VAL
submission_checklist.md           ← VAL
```

# Shared Ownership (VAL Finalizes)
```
README.md                         ← ALL (VAL has final edit)
guardrails.md                     ← ALL (VAL enforces)
mistakes.md                       ← ALL (VAL reviews)
```

---

# Validator's Parallel Work Map (Never Idle)

| Phase | While Coders Build... | Validator Does... |
|---|---|---|
| **Phase 0** | Scaffold + Colab skeleton | Prepare review prompts, pre-check architecture against all 9 MSRs, set up W&B project |
| **Phase 1** | Environment core | Prepare environment sanity test suite (`test_environment.py`), verify OpenEnv inheritance, write `test_episode.py` skeleton |
| **Phase 2** | Training pipeline | Write `test_reward.py`, prepare reward function review checklist, verify logging hooks template |
| **Phase 3** | Training run | Verify training run is real (not mocked), validate W&B data is flowing, prepare plot review criteria |
| **Phase 4** | HF Space + README | Verify HF Space publicly accessible (incognito), run Colab top-to-bottom, review README completeness |
| **Phase 5** | Blog/video | Review writeup from non-technical judge perspective, verify all links, check video length |
| **Phase 6** | — | Run final submission validation sequence |

---

# Merge Gate Procedures

## At Every Gate — Standard Procedure

### 1. AI Code Review Prompt (paste-ready)
```
Review this code for The Examiner hackathon project. Check for:
1. OpenEnv compliance: inherits from Environment base class, async methods, StepResult return type
2. Reward function: matches pseudocode (R = 0.70*acc - 0.50*FA + 0.20*eff + 0.10*diag), no LLM judge, no constants
3. MSR compliance: [list specific MSRs for this gate]
4. Code quality: no unused imports, no print debugging, type hints present, no hardcoded hyperparameters
5. Scope: no out-of-scope features (check against guardrails.md Section 3)

Return findings as: PASS/FAIL for each check. For failures, specify exact line and fix.
```

### 2. MSR Gate Check
For each gate, check the specific MSRs that should be satisfied:
- [ ] Which MSRs should be closed by this merge?
- [ ] For each: is it actually closed? Evidence?
- [ ] Update guardrails.md MSR checklist

### 3. Judging Criterion Impact Check
- Does this merge advance ENV_INNOV? How?
- Does this merge advance STORYTELLING? How?
- Does this merge risk any criterion? How?

---

## Phase 0 Gate — Foundation
**MSRs to verify:** MSR-1 (partial), MSR-2 (partial)

**Checks:**
- [ ] `examiner_env/` scaffold exists with OpenEnv structure
- [ ] `from openenv.core import Environment` imports successfully
- [ ] `openenv.yaml` is present and valid
- [ ] Colab notebook has install + import cells
- [ ] W&B init cell references project 'the-examiner'
- [ ] KB has exactly 10 sections with substantive content

**Time budget:** 15 min

---

## Phase 1 Gate — Environment Core
**MSRs to verify:** MSR-1 ✅ (full)

**RL-Specific Manual Checks:**
- [ ] `ExaminerEnvironment` inherits from `openenv.core.Environment` (not reimplemented)
- [ ] All environment methods are `async def`
- [ ] `step()` returns `StepResult` (not tuple, not dict)
- [ ] Reward function matches pseudocode from architecture.md exactly
  - [ ] 4 components, correct weights (0.70, -0.50, 0.20, 0.10)
  - [ ] No LLM judge in reward computation
  - [ ] No constant return values
- [ ] Student has 2 distinct prompts (genuine vs faking)
- [ ] Partition sampling: k ∈ {3, 4, 5}, randomly assigned
- [ ] Force-termination at turn 20 with -0.5 penalty
- [ ] `test_episode.py` runs end-to-end: reset → Ask × N → Classify → reward

**AI Review Prompt for Phase 1:**
```
Review examiner_env/server/examiner_environment.py. Verify:
1. Class inherits from openenv.core.Environment
2. reset() is async, returns ExaminerObservation
3. step() is async, handles Ask and Classify, returns StepResult
4. Force-termination at turn 20
5. Partition is randomly sampled (k from {3,4,5})
Paste the file contents below:
[PASTE FILE]
```

**Time budget:** 30 min

---

## Phase 2 Gate — Training Pipeline
**MSRs to verify:** MSR-2 ✅, MSR-3 (started)

**Checks:**
- [ ] Training script uses `GRPOTrainer` (search for the class name)
- [ ] No custom training loop
- [ ] All hyperparameters in `TrainingConfig` dataclass (no hardcoded values)
- [ ] W&B logging present: reward, accuracy, false_accusations, efficiency, turns, loss, kl
- [ ] Colab notebook runs top-to-bottom without error (test in Colab)
- [ ] No local file paths in Colab notebook
- [ ] Smoke test: post-training reward > pre-training reward
- [ ] Reward unit tests pass (all 6 cases in `test_reward.py`)

**Integration Check:**
- [ ] C2's training script correctly imports C1's `ExaminerEnvironment`
- [ ] Reward function called by training pipeline = same function in `reward.py`

**Time budget:** 30 min

---

## Phase 3 Gate — Training Evidence
**MSRs to verify:** MSR-3 ✅

**Critical Verification — Is the Training Run REAL?**
- [ ] W&B dashboard shows timestamped data from hackathon timeframe
- [ ] Episode count matches expected (target: 500)
- [ ] Reward curve shows 3 visible phases (not a straight line, not random)
- [ ] Loss values are finite (no NaN, no constant)
- [ ] Plots in `outputs/plots/` match W&B data (not hand-drawn, not synthetic)

**Transcript Verification:**
- [ ] `before.txt` shows definitional questions and incorrect classification
- [ ] `after.txt` shows mechanistic/causal questions and correct classification
- [ ] The contrast between before/after is immediately obvious

**Time budget:** 20 min

---

## Phase 4 Gate — Deployment
**MSRs to verify:** MSR-5 ✅, MSR-6 ✅, MSR-7 ✅, MSR-8 (partial), MSR-9 ✅

**HF Space Verification:**
- [ ] Open HF Space URL in **incognito browser**
- [ ] Gradio interface loads without errors
- [ ] "Live Demo" tab: click "Run Episode" → transcript appears
- [ ] "Training Results" tab: plots visible, transcripts visible
- [ ] "About" tab: 3-sentence narrative present

**README Verification:**
- [ ] 3-sentence narrative present at top
- [ ] Problem statement section exists
- [ ] Environment description section exists
- [ ] Training approach section exists
- [ ] Results section has embedded plots (not placeholders)
- [ ] HF Space link present and clickable (MSR-7)
- [ ] All material links present (MSR-8 — blog may still be pending)
- [ ] Setup instructions with exact commands

**MSR-9 Check:**
- [ ] `git ls-files | grep -E '\.(mp4|avi|mov|webm|mkv)$'` returns nothing

**Time budget:** 20 min

---

## Phase 5 Gate — Storytelling
**MSRs to verify:** MSR-4 ✅, MSR-8 ✅

**Blog/Video Review (Judge Perspective):**
- [ ] 3-sentence narrative appears verbatim
- [ ] Non-technical person can understand the problem and solution
- [ ] Reward curve is embedded and shows improvement
- [ ] Before/after transcripts are shown
- [ ] Video is under 2 minutes (if video chosen)
- [ ] No unexplained jargon
- [ ] Link to HF Space is present

**README Final Link Check:**
- [ ] HF Space link → opens live Space
- [ ] Blog/video link → opens published artifact
- [ ] Training notebook link → opens Colab
- [ ] All links return 200 (not 404)

**Time budget:** 15 min

---

# Final Submission Validation Sequence (Last 30 Minutes)

Execute in this exact order. Do not skip any step.

```
1. ☐ Open guardrails.md → Run full MSR checklist (Section 2) → All 9 checked?
2. ☐ For each judging criterion, state strongest evidence:
     ENV_INNOV (40%): "Our environment is the first to train diagnostic question design under adversarial deception"
     STORYTELLING (30%): "3-sentence narrative + before/after transcript contrast + reward curve"
     REWARD_EVIDENCE (20%): "Real W&B plots showing 3-phase training curve"
     PIPELINE (10%): "GRPO + Unsloth + TRL, runs in Colab, 20-line reward function"
3. ☐ Open HF Space in incognito → loads? → run episode? → results tab works?
4. ☐ Open Colab notebook → Kernel > Restart and Run All → completes without error?
5. ☐ Open README → are all 9 MSR-relevant items present and linked?
6. ☐ Check HF Hub repo → git ls-files → no .mp4/.avi/.mov/.webm?
7. ☐ Read blog post/watch video → under 2 min? → non-technical-audience ready?
8. ☐ View reward/loss plots → real run? → shows improvement?
9. ☐ AI self-audit: paste full codebase summary + ask "Does this submission satisfy all 9 MSRs and all 4 judging criteria?"
10. ☐ All 3 team members confirm → SUBMIT
```

---

# Bug Triage Protocol

## Severity Classification
| Severity | Definition | Action |
|---|---|---|
| 🔴 Blocker | Breaks an MSR or judging criterion | Drop everything. Fix immediately. |
| 🟡 Degraded | Works but sub-optimal (e.g., ugly plot, slow training) | Fix if time permits. Log in mistakes.md. |
| 🟢 Minor | Cosmetic (typo, formatting) | Fix during Phase 5 polish. |

## AI Diagnosis Prompt
```
Bug report:
- File: [filename]
- Error: [paste error message]
- Expected behavior: [what should happen]
- Actual behavior: [what happens]
- MSR at risk: [MSR-N or NONE]

Context: This is for The Examiner, an OpenEnv RL environment that trains diagnostic question design. The environment inherits from openenv.core.Environment. Training uses Unsloth + TRL GRPOTrainer.

Diagnose the root cause and provide a minimal fix. Do not change the reward function formula or environment interface.
```

## Bug Report Template for mistakes.md
```
### Bug [session]-[number]
- **Severity:** 🔴/🟡/🟢
- **File:** [path]
- **Error:** [one-line description]
- **MSR at risk:** [MSR-N or NONE]
- **Judging criterion impacted:** [criterion or NONE]
- **Root cause:** [description]
- **Fix:** [what was changed]
- **Corrected prompt:** [the AI prompt that produced the fix]
- **Re-validation:** [how we confirmed the fix works]
```

## "Second AI Opinion" Protocol
Trigger conditions (use a second model to verify):
1. Reward logic is disputed between C1 and C2
2. OpenEnv integration behavior is uncertain
3. Training results look anomalous (e.g., reward decreasing, sudden spikes)

Process: Paste the disputed code + context into a **different AI model** (e.g., if using Claude, check with Cursor's model). Compare outputs. If they disagree, go with the output that matches the architecture.md specification.

---

# Validator Tests to Write

## test_environment.py
```
Test OpenEnv compliance:
1. ExaminerEnvironment inherits from Environment
2. reset() returns ExaminerObservation with 5 fields
3. step(Ask) returns StepResult with reward=0 (step reward) and done=False
4. step(Classify) returns StepResult with done=True and finite reward
5. Force-termination at turn 20 returns reward=-0.5
6. Partition has k sections KNOWS where k in {3,4,5}
```

## test_reward.py
```
Test reward function:
1. Perfect classification → reward > 0.8
2. All-FAKING guess → reward strongly negative (< -1.0)
3. All-KNOWS guess → reward near 0
4. Same accuracy, fewer turns → higher reward (efficiency)
5. Each false accusation → -0.5 penalty
6. Classify at turn 1 vs turn 20 → efficiency difference of ~0.19
```

## test_episode.py
```
Full episode smoke test:
1. reset() + 5 Ask actions + Classify → episode completes
2. Reward is finite float (not NaN, not inf)
3. get_metrics() returns dict with 4 keys
4. render() returns non-empty string
5. question_history has 5 entries after 5 Ask actions
```
