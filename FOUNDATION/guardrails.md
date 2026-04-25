# 🛡️ GUARDRAILS.md — THE EXAMINER
## Supreme Authority Document · OpenEnv Hackathon India 2026
> **This file supersedes all other documents in any conflict. Every team member and their AI tool reads this first, every session.**

---

# Section 1: Judging Criteria Alignment Matrix

## 1.1 Environment Innovation (40%)

**Novelty Claim:** The Examiner is the first RL environment that trains *diagnostic question design under adversarial deception*. No prior environment teaches an agent to formulate questions whose answers differ based on genuine vs. faked knowledge. DataEnvGym trains difficulty calibration — fundamentally different. This is verifiable novelty.

**Design Decisions That Maximize This Score:**
- Two-agent adversarial structure (Examiner vs. Student) with hidden knowledge partition
- Student has two distinct response modes (genuine/faking) — the environment's core mechanic
- Belief scratchpad as working memory — enables observable reasoning strategy evolution
- Question-type hierarchy emerges from training, not hard-coded (definitional → mechanistic → causal counterfactual)
- Programmatic reward with zero LLM judges — clean signal, reproducible, verifiable
- Adaptive Student difficulty (Snorkel AI bonus) — environment requirements change during training

**What "Innovative Enough" Looks Like:**
- ✅ NOVEL: An agent that learns which question *types* separate real knowledge from faking — no prior RL env does this
- ✅ NOVEL: Adversarial deception detection as an RL objective
- ✅ NOVEL: Belief-updating under partial observability with a text-based scratchpad
- ❌ DERIVATIVE: A quiz-bot that asks harder questions (just difficulty scaling — DataEnvGym already exists)
- ❌ DERIVATIVE: A fact-checking environment (entailment verification — not diagnostic reasoning)
- ❌ DERIVATIVE: An environment that just grades student answers (that's QA, not diagnostic design)

**AI Prompt Guardrail:** If any AI-generated environment mechanic resembles CartPole, MuJoCo, Atari, or standard text-QA benchmarks → **FLAG FOR REVIEW** before committing. The Examiner's novelty is in *question design under deception*, not in question-answering.

**Innovation Re-validation Checkpoints:**
- [ ] Phase 0: Confirm environment architecture matches this novelty claim
- [ ] Phase 1: After core environment is built, re-read this section — does the implementation still deliver the novelty?
- [ ] Phase 3: After training, verify the agent learned diagnostic question design (not domain-specific shortcuts)
- [ ] Phase 5: Writeup/video explicitly articulates the novelty claim from this section

---

## 1.2 Storytelling & Presentation (30%)

**3-Sentence Narrative (verbatim in README, writeup, video, HF Space description):**
> LLMs are dangerously good at sounding like they know things they don't — and current AI systems cannot tell the difference. The Examiner is an RL environment that trains an AI to design diagnostic questions that reliably separate genuine knowledge from confident faking. After training, the Examiner learns to skip surface-level questions and ask the mechanistic, causal questions that only genuine experts can answer.

**What a Non-Technical Judge Must Understand After the Demo:**
1. The problem: LLMs (and humans) can fake expertise convincingly
2. The solution: An AI examiner that *learns* which questions expose fakers
3. The proof: Side-by-side transcripts (episode 10 vs. episode 400) + reward curve going up

**Presentation Artifacts:**
| Artifact | Format | Owner | Due |
|---|---|---|---|
| README with results | Markdown | ALL (VAL finalizes) | Phase 4 |
| HuggingFace mini-blog | HF blog post | VAL | Phase 5 |
| Demo video (<2 min) | YouTube URL | VAL | Phase 5 |
| HF Space with Gradio | Live app | C2 | Phase 4 |

**Story Impact Test:** At every implementation decision, ask: *"Does this make the story harder or easier to tell?"* If harder → reconsider the approach. Complexity the judge can't see in 60 seconds is wasted complexity.

**Storytelling is Built Continuously:**
- Phase 1: Screenshot environment output, save example transcripts
- Phase 3: Capture before/after transcripts, export reward curve plots
- Phase 4: README written with real results, not placeholders
- Phase 5: Blog/video/slides assembled from Phase 1-4 artifacts — NOT created from scratch

---

## 1.3 Showing Improvement in Rewards (20%)

**Training Evidence We Will Produce:**
| Evidence | Format | Source | Location |
|---|---|---|---|
| Reward curve (episodes vs. mean reward) | PNG plot | W&B export | `outputs/plots/reward_curve.png` |
| Classification accuracy curve | PNG plot | W&B export | `outputs/plots/accuracy_curve.png` |
| Before transcript (episode ~10) | Text | Training logs | `outputs/transcripts/before.txt` |
| After transcript (episode ~400) | Text | Training logs | `outputs/transcripts/after.txt` |
| Question type distribution shift | PNG plot | Analysis script | `outputs/plots/question_types.png` |

**Baseline Agent:** A random examiner that asks definitional questions and classifies randomly. Implemented in Phase 0. Expected baseline accuracy: ~50% (chance). Expected baseline reward: ~-0.1.

**Non-Negotiable:** Plots are from **real training runs**. Not generated. Not approximated. Not mocked. MSR-3 is a hard gate.

**Reward Logging Verification:** Before any training run is considered valid, confirm:
- [ ] W&B (or equivalent) is receiving reward signals per episode
- [ ] Loss is logged per gradient update
- [ ] A test run of 5 episodes produces a visible data point in the dashboard

---

## 1.4 Reward & Training Pipeline (10%)

**Reward Function in Plain English:**
1. **Classification accuracy (70%):** For each of 10 sections, +1 if correctly classified (KNOWS→KNOWS or FAKING→FAKING), -1 if false positive (called FAKING when truly KNOWS), 0 if missed a faker. Normalized to [-1, 1].
2. **False accusation penalty (-0.5 per error):** Each section incorrectly classified as FAKING when student truly KNOWS incurs a -0.5 penalty. Encourages precision.
3. **Efficiency bonus (20%):** Bonus = (20 - turns_used) / 20 × 0.2. Rewards faster correct classification.
4. **Diagnostic quality bonus (10%):** For each question on a FAKING section, did the question surface distinguishably different behavior? Proxy: answer divergence from genuine-mode baseline.

**Final formula:**
```
R = 0.70 × accuracy_score − 0.50 × false_accusations + 0.20 × efficiency_bonus + 0.10 × diagnostic_quality_bonus
```

**"Coherent Reward Logic" for Our Environment:** The reward must incentivize the Examiner to (a) ask fewer, more diagnostic questions, (b) correctly identify fakers without falsely accusing genuine knowers, and (c) submit classification before the turn limit. Every component serves one of these three goals.

**Pipeline Smoke Test (minimum run proving non-trivial improvement):**
- 20 episodes with a random policy → log mean reward
- 20 episodes after 50 training steps → log mean reward
- If mean reward after training > mean reward before → pipeline produces non-trivial improvement
- If not → pipeline is broken. Do not proceed to Phase 3.

**Even at 10% weight, a broken pipeline fails MSR-2 and MSR-3. Treat it as a blocker.**

---

# Section 2: Minimum Submission Requirements Checklist

> 🚨 **Every MSR is a hard disqualifier. Partial compliance = non-compliance. This checklist is re-run at every merge gate.**

- [ ] 🚨 **MSR-1** | OpenEnv integration confirmed | Phase: 0-1 | Owner: C1 | Verified: VAL at Phase 1 gate | **How to verify:** `ExaminerEnv` inherits from OpenEnv `Environment` base class. `from openenv.core import Environment` in imports. `reset()`, `step()`, `state()` follow OpenEnv API.
- [ ] 🚨 **MSR-2** | Training script + Colab notebook runnable | Phase: 2-3 | Owner: C2 | Verified: VAL at Phase 3 gate | **How to verify:** Open Colab notebook in incognito. Click "Run All". No errors. No local path assumptions. Training completes and logs to W&B.
- [ ] 🚨 **MSR-3** | Real loss + reward plots from actual run | Phase: 3 | Owner: C2 | Verified: VAL at Phase 3 gate | **How to verify:** Plots exist in `outputs/plots/`. W&B dashboard shows matching data. Timestamps on logs match hackathon timeframe. Three visible phases in reward curve.
- [ ] 🚨 **MSR-4** | Writeup/video/slides published and linked | Phase: 5 | Owner: VAL | Verified: VAL at Phase 6 | **How to verify:** HF blog post URL works. Content matches 3-sentence narrative. Shows reward curve + before/after transcripts. Under 2 minutes if video.
- [ ] 🚨 **MSR-5** | HuggingFace Space live and runnable | Phase: 4 | Owner: C2 | Verified: VAL at Phase 4 gate | **How to verify:** Open HF Space URL in incognito browser. Gradio interface loads. Can run a live episode. See environment output.
- [ ] 🚨 **MSR-6** | README complete (problem + env + results) | Phase: 4-5 | Owner: ALL (VAL finalizes) | Verified: VAL at Phase 6 | **How to verify:** README has: problem statement, environment description, training approach, results with plots, setup instructions. No placeholder sections.
- [ ] 🚨 **MSR-7** | README links to HF Space | Phase: 4 | Owner: VAL | Verified: VAL at Phase 6 | **How to verify:** Click the HF Space link in README. It opens the live Space.
- [ ] 🚨 **MSR-8** | README links to all materials | Phase: 5 | Owner: VAL | Verified: VAL at Phase 6 | **How to verify:** README has clickable links to: HF Space, blog/video/slides, training notebook. All links resolve.
- [ ] 🚨 **MSR-9** | No video files in HF Hub repo | Phase: 4-5 | Owner: ALL | Verified: VAL at Phase 6 | **How to verify:** `git ls-files` on HF repo shows no .mp4, .avi, .mov, .webm files. Videos are YouTube/external URLs only.

---

# Section 3: Scope Guardrails

## IN Scope (Exhaustive)
- ExaminerEnv class inheriting from OpenEnv `Environment`
- ML Theory knowledge base with 10 sections
- Student agent with genuine/faking modes (scripted LLM, Llama-3B-Instruct)
- Examiner agent (Qwen2.5-7B or Llama-3.1-8B, trained with GRPO via Unsloth)
- Observation space: KB section titles, question history, turn counter, belief scratchpad, remaining turns
- Action space: Ask(section_id, question_text) and Classify(partition_dict)
- 4-component reward function (accuracy, false accusation penalty, efficiency, diagnostic quality)
- Optional step-level intermediate reward (information gain proxy, disabled after 200 episodes)
- Training script in Google Colab using Unsloth + TRL GRPOTrainer
- W&B logging for reward, loss, accuracy, false accusation rate, efficiency, mean turns
- Reward curve + accuracy curve + question type distribution plots
- Before/after transcript selection
- Gradio interface on HuggingFace Space for live episode playback
- README with all required links and content
- HF mini-blog post OR <2 min YouTube video
- Adaptive Student difficulty (Snorkel AI bonus — Phase 3+)
- Domain curriculum transfer test (ML theory → database fundamentals, Phase 3+ if time permits)

## OUT of Scope (Exhaustive — Reject if AI Generates)
- Multiple knowledge domains simultaneously in v1 (one domain at a time)
- Student as a separately RL-trained model (v2 — only if base is complete)
- Web UI beyond Gradio on HF Space
- Mobile interface
- User accounts or authentication
- Real-time multiplayer
- Integration with external APIs beyond HF Hub and W&B
- Custom visualization dashboards (use W&B exports)
- Automated hyperparameter search (manual config only)
- Model distillation or quantization beyond what Unsloth provides
- Multi-GPU training orchestration
- Leaderboard system
- Any evaluation using LLM-as-judge

## 🚨 Scope Creep Triggers (RL/OpenEnv Specific)
| Trigger | Why It's Dangerous | Action |
|---|---|---|
| AI adding observation spaces we didn't define | Breaks OpenEnv interface contract, adds complexity | Reject. Use only the 5 observation components defined above |
| AI expanding the action space "for completeness" | Only Ask and Classify exist. Anything else is scope creep | Reject. Two action types only |
| AI generating a custom training loop instead of using Unsloth/TRL | Violates MSR-2 directly | 🚨 Reject immediately |
| AI reimplementing OpenEnv internals instead of inheriting | Violates MSR-1 directly | 🚨 Reject immediately |
| AI generating extra environment variants "for ablation" | Time sink. One environment, one domain for v1 | Reject unless Phase 3 is complete |
| AI adding a web UI when we only asked for a Colab notebook | Scope creep, time waste | Reject |
| AI adding evaluation metrics beyond what's in the reward function | Noise. The reward function IS the evaluation | Reject unless it's for presentation plots |

**Quarantine Rule:** Any generated code outside the IN scope list is quarantined in a `_quarantine/` folder and reviewed before commit. Never commit quarantined code directly.

---

# Section 4: Tech Stack Guardrails

## Locked, Approved Tech Stack

| Component | Technology | Version | MSR | Non-Negotiable |
|---|---|---|---|---|
| Environment Framework | OpenEnv (`openenv-core`) | latest from `pip install openenv-core` | MSR-1 | ✅ |
| RL Training | Unsloth + HuggingFace TRL `GRPOTrainer` | `unsloth>=2025.4`, `trl>=0.15` | MSR-2 | ✅ |
| Base Model (Examiner) | Qwen2.5-7B-Instruct OR Llama-3.1-8B-Instruct | HF latest | MSR-2 | ✅ |
| Student Model | Llama-3.2-3B-Instruct (or equivalent small model) | HF latest | — | ✅ |
| Training Environment | Google Colab (T4/A100 GPU) | — | MSR-2 | ✅ |
| Experiment Logging | Weights & Biases (`wandb`) | `wandb>=0.19` | MSR-3 | ✅ |
| Deployment | HuggingFace Spaces + Gradio | `gradio>=5.0` | MSR-5 | ✅ |
| Model/Data Hosting | HuggingFace Hub | `huggingface_hub>=0.27` | MSR-5 | ✅ |
| Python | 3.10+ | 3.10 | — | ✅ |
| Serialization | Pydantic (via OpenEnv) | v2 | MSR-1 | ✅ |

## Explicitly Banned Approaches
- ❌ Custom training loops that bypass `GRPOTrainer` from TRL
- ❌ Reimplementing `Environment` base class instead of inheriting from `openenv.core`
- ❌ Local-only training runs with no W&B logging
- ❌ Video files committed to HF Hub repo (MSR-9)
- ❌ LLM-as-judge in reward computation
- ❌ Any entailment model or semantic similarity in the reward function
- ❌ Training on local machine instead of Colab (judges must be able to re-run)
- ❌ `print()` debugging left in production code
- ❌ Hardcoded API keys in any file

## ⚠️ AI Hallucination Warning
AI tools frequently suggest outdated HuggingFace API patterns. Specific risks:
- `push_to_hub()` API has changed — verify against current `huggingface_hub` docs
- TRL `GRPOTrainer` constructor args change between versions — always check `GRPOConfig` current signature
- OpenEnv uses `async` by default — AI may generate sync-only code
- `gradio.Interface` vs `gradio.Blocks` — use `Blocks` for the HF Space (more control)
- Unsloth `FastLanguageModel.from_pretrained()` parameters — verify against unsloth.ai docs

---

# Section 5: AI Tool Usage Rules

## Approved Tool-to-Task Mapping
| Task | Primary Tool | Secondary Tool |
|---|---|---|
| Architecture/design decisions | Claude | — |
| Code implementation (environment) | Cursor / Claude | Copilot for autocomplete |
| Code implementation (training script) | Cursor / Claude | — |
| Code review | Claude | — |
| Colab notebook iteration | Colab AI + Claude | — |
| Debugging | Cursor / Claude | — |
| README/blog writing | Claude | — |
| Git operations | Manual | — |

## Prompting Rules
- **Maximum 2 re-prompt attempts** per task before escalating to Validator. Log failure in `mistakes.md`.
- **Minimal Context, Maximum Precision:** Never dump the full codebase. Include only: the file being modified, its imports, and the relevant interface.
- **RL-Specific Context Hygiene:** When prompting for any training-related code, ALWAYS include:
  - Current `ExaminerEnv` class signature
  - Current reward function
  - Current observation/action space dataclass definitions
- **OpenEnv-Specific Prompt Rule:** ALWAYS paste the relevant OpenEnv base class interface (`Environment`, `Action`, `Observation`, `StepResult`) into context when asking AI to implement environment methods. **AI WILL hallucinate the interface otherwise.**
- **Confident ≠ Correct:** The sanity check wins, not the AI's tone. If AI says "this is correct" but the sanity check fails, the AI is wrong.

---

# Section 6: Code Quality Non-Negotiables (RL/ML Context)

1. **No hardcoded hyperparameters.** All hyperparameters in a `TrainingConfig` dataclass at the top of the training script.
2. **No training runs without logging.** Every run emits loss and reward to W&B. No exceptions (MSR-3).
3. **No placeholder reward functions committed.** Reward logic must be real and coherent from first commit. A reward function returning constants is a 🔴 blocker.
4. **No silent training failures.** The training script must surface errors with tracebacks, not silently continue with broken rewards or NaN losses.
5. **No `print()` debugging left in environment code.** Use Python `logging` module.
6. **No unused imports.** AI generates these constantly. Remove before commit.
7. **Colab notebook must run top-to-bottom** with no modifications by a judge (MSR-2). Test this explicitly before Phase 3 gate.
8. **HF Space must be publicly accessible** before final submission. Verify with incognito browser (MSR-5).
9. **All reward components must be tested independently** with hand-crafted episodes before integration.
10. **Type hints on all function signatures.** OpenEnv uses Pydantic — type safety is mandatory.

---

# Section 7: Git Non-Negotiables

1. **No direct commits to `main`.** All work on feature branches.
2. **No force pushes.** Ever. Under any circumstances.
3. **No cross-ownership AI generation.** C1's AI does not generate files owned by C2, and vice versa.
4. **Structured commit messages:**
   ```
   [type] C1|C2|VAL | stage-[id] | [feature] | [AI tool] | MSR:[n,n] | [passes/fails sanity]
   ```
   Types: `feat`, `fix`, `integrate`, `validate`, `docs`, `config`, `train`, `deploy`
5. **No commits without passing all 3 sanity checks** for the changed files.
6. **No merges without Validator gate clearance.**
7. **Branch naming:** `feat/c1-S[stage]-[feature]`, `feat/c2-S[stage]-[feature]`, `validate/phase-[n]`

---

# Section 8: Storytelling Non-Negotiables (30% of Score)

1. The **3-sentence project narrative** from Section 1.2 must appear **verbatim** in:
   - README.md (top of file)
   - HF Space description
   - Blog post / video script / slides opening
2. **Every demo must show, in this order:**
   1. The environment running (Gradio interface)
   2. The agent BEFORE training (episode ~10 transcript — definitional questions, wrong classification)
   3. The agent AFTER training (episode ~400 transcript — mechanistic questions, correct classification)
   4. The reward curve (episodes vs. mean reward, three visible phases)
3. The writeup/video/slides must be **completable by a non-technical person who reads only the README.**
4. Presentation artifacts are produced **during the hackathon** — they are assigned stages in the implementation plan, not afterthoughts.
5. **No jargon without explanation.** Every technical term in the writeup gets a one-sentence plain-English explanation immediately after.
   - Example: "GRPO (Group Relative Policy Optimization) — a training method that improves the model by comparing different responses to the same prompt."

---

# Section 9: Session Start Ritual (Mandatory)

```
1. git pull origin main
2. Read "Recurring Mistakes Index" in mistakes.md
3. Read "MSR Checklist" in guardrails.md — note which MSRs are still open
4. Confirm your file ownership list is current (check project_structure.md)
5. Check current phase and stage from your implementation file
6. Paste context_primer.md into your AI tool
7. Begin work — not before
```

---

# Section 10: Session End Ritual (Mandatory)

```
1. Run all 3 sanity checks on everything generated this session
2. Re-check MSR checklist — did this session close any MSRs? Update checkboxes above
3. Log any mistakes in mistakes.md
4. Commit with structured message format from Section 7
5. Update progress markers in your implementation file
6. If at merge gate — send handoff message using the exact template from your implementation file
```

---

# Section 11: Guardrails Versioning

> Any change to `guardrails.md` requires all 3 team members to acknowledge before work resumes.
> Changes to scope, stack, or submission criteria propagate to all affected documents immediately.

### Changelog
| Date | Author | Change | Acknowledged By |
|---|---|---|---|
| 2026-04-25 | ALL | v1.0 — Initial guardrails created | C1, C2, VAL |
