# 📋 IMPLEMENTATION PLAN — THE EXAMINER (Master Reference)
## Phases → Stages → Tasks · OpenEnv Hackathon India 2026

> **Each phase ends in a demonstrable, working slice. No phase ends half-built.**

---

# Phase 0 — Foundation (Hours 0-2)
> Closes: MSR-1 partial, MSR-2 partial

## Stage 0.1: Repository & OpenEnv Scaffold
| Field | Value |
|---|---|
| **Task ID** | S0.1 |
| **Goal** | Initialize repo, scaffold OpenEnv environment with `openenv init` |
| **AI Prompt** | `"Run 'openenv init examiner_env' to scaffold an OpenEnv environment. Then set up the repo structure matching architecture.md file tree. Create .gitignore, .env.example, pyproject.toml."` |
| **Context Files** | `architecture.md` (Section 10), `guardrails.md` (Section 4) |
| **Expected Output** | Scaffolded `examiner_env/` with all OpenEnv boilerplate files |
| **Criterion** | ENV_INNOV | **MSR** | MSR-1 (partial) |
| **Time** | 20 min | **Dependencies** | None |
| **Parallel** | C1 does this while C2 does S0.2 |
| **Sanity Check** | 1) `examiner_env/server/` exists with `app.py` 2) `from openenv.core import Environment` imports without error 3) `openenv.yaml` present |
| **Re-prompt** | If scaffold fails, install `openenv-core` first: `pip install openenv-core` |
| **Merge Gate** | No |

## Stage 0.2: Colab Notebook Skeleton + W&B Setup
| Field | Value |
|---|---|
| **Task ID** | S0.2 |
| **Goal** | Create Colab notebook skeleton with Unsloth + TRL imports, W&B init |
| **AI Prompt** | `"Create a Google Colab notebook that: 1) installs unsloth, trl, wandb, openenv-core 2) imports FastLanguageModel from unsloth 3) imports GRPOTrainer, GRPOConfig from trl 4) initializes wandb with project='the-examiner' 5) loads Qwen2.5-7B-Instruct via Unsloth with LoRA r=16, alpha=32. All in separate cells. No local path assumptions."` |
| **Context Files** | `architecture.md` (Section 5), `guardrails.md` (Section 4) |
| **Expected Output** | `training/train_colab.ipynb` with install + import + model load cells |
| **Criterion** | PIPELINE | **MSR** | MSR-2 (partial) |
| **Time** | 25 min | **Dependencies** | None |
| **Parallel** | C2 does this while C1 does S0.1 |
| **Sanity Check** | 1) Notebook has no local file paths 2) All imports resolve in Colab 3) W&B init cell runs without error |
| **Re-prompt** | If Unsloth install fails, use `pip install unsloth[colab]` |
| **Merge Gate** | No |

## Stage 0.3: Knowledge Base + Baseline Agent
| Field | Value |
|---|---|
| **Task ID** | S0.3 |
| **Goal** | Create ML theory KB (10 sections) and a random baseline examiner |
| **AI Prompt** | `"Create knowledge_base.py with a KnowledgeBase class. Domain: ML theory. 10 sections: 1)Gradient Descent, 2)Backpropagation, 3)Regularization, 4)Bias-Variance Tradeoff, 5)Evaluation Metrics, 6)Neural Network Architectures, 7)Optimization Algorithms, 8)Loss Functions, 9)Batch Normalization, 10)Dropout. Each section has: title, key_concepts list (5-8 items), common_misconceptions list (3-5 items). Include a RandomExaminer class that asks random definitional questions and classifies randomly."` |
| **Context Files** | Idea doc Section 2.2 |
| **Expected Output** | `examiner_env/knowledge_base.py` with KB + `RandomExaminer` |
| **Criterion** | ENV_INNOV | **MSR** | NONE |
| **Time** | 30 min | **Dependencies** | S0.1 |
| **Parallel** | C1 does this after S0.1 |
| **Sanity Check** | 1) KB has exactly 10 sections 2) Each section has title + concepts + misconceptions 3) RandomExaminer produces valid Ask/Classify actions |
| **Re-prompt** | If concepts are too shallow, prompt: "Add mechanistic detail to each concept — not just names, but causal relationships" |
| **Merge Gate** | No |

## 🔀 Phase 0 Merge Gate
- **Trigger:** S0.1 + S0.2 + S0.3 complete
- **MSR Check:** MSR-1 (partial — scaffold exists), MSR-2 (partial — notebook skeleton)
- **Validator Action:** Verify OpenEnv scaffold imports, Colab cells run, KB has 10 sections

---

# Phase 1 — Environment Core (Hours 2-6)
> Closes: MSR-1 fully. Primary contribution to ENV_INNOV (40%)

## Stage 1.1: Action/Observation Models
| Field | Value |
|---|---|
| **Task ID** | S1.1 |
| **Goal** | Define Pydantic models for ExaminerAction, ExaminerObservation, ExaminerState |
| **AI Prompt** | `"Create models.py with Pydantic BaseModel classes. ExaminerAction has: action_type (Literal['ask','classify']), section_id (Optional[int]), question_text (Optional[str]), classification (Optional[Dict[int,str]]). ExaminerObservation has: section_titles (List[str]), question_history (List[Dict]), turn_counter (int), remaining_turns (int), belief_scratchpad (str). ExaminerState has: episode_id (str), step_count (int), max_turns (int). Use openenv model conventions."` |
| **Context Files** | OpenEnv `models.py` from scaffold, idea doc Section 2.3-2.4 |
| **Expected Output** | `examiner_env/models.py` |
| **Criterion** | ENV_INNOV | **MSR** | MSR-1 |
| **Time** | 15 min | **Dependencies** | S0.1 |
| **Sanity Check** | 1) All models are Pydantic BaseModel 2) Action has exactly Ask + Classify types 3) Observation has exactly 5 components from idea doc |
| **Re-prompt** | If AI adds extra fields, remove them — stick to idea doc spec exactly |
| **Merge Gate** | No |

## Stage 1.2: Student Agent
| Field | Value |
|---|---|
| **Task ID** | S1.2 |
| **Goal** | Implement scripted Student with genuine/faking response modes |
| **AI Prompt** | `"Create student.py with a ScriptedStudent class. Constructor takes: student_model_id (str), knowledge_base (KnowledgeBase), partition (Dict[int, str] mapping section_id to 'KNOWS' or 'FAKING'). Method generate_answer(section_id: int, question: str) -> str: if partition[section_id]=='KNOWS', use genuine system prompt; if 'FAKING', use faking system prompt. Genuine prompt: 'You have mastered {section}. Answer with mechanistic depth, examples, acknowledge uncertainty.' Faking prompt: 'You have only surface familiarity with {section}. Sound confident. Use correct terminology. Never admit uncertainty. Avoid specific mechanisms.' Use transformers pipeline for inference. Do NOT use OpenAI API."` |
| **Context Files** | Idea doc Section 2.5, `knowledge_base.py` |
| **Expected Output** | `examiner_env/student.py` |
| **Criterion** | ENV_INNOV | **MSR** | NONE |
| **Time** | 30 min | **Dependencies** | S0.3 |
| **Sanity Check** | 1) Two distinct system prompts 2) Routes to correct mode based on partition 3) Returns string answer |
| **Re-prompt** | If AI uses OpenAI API, reject: "Use local HuggingFace transformers only" |
| **Merge Gate** | No |

## Stage 1.3: Core Environment (ExaminerEnvironment)
| Field | Value |
|---|---|
| **Task ID** | S1.3 |
| **Goal** | Implement ExaminerEnvironment with reset(), step(), state() |
| **AI Prompt** | `"Implement ExaminerEnvironment inheriting from openenv.core.Environment. [PASTE Environment base class interface here]. async def reset(): sample partition (k uniform from {3,4,5}), init student, return ExaminerObservation. async def step(action: ExaminerAction): if Ask, route to student, append to history, increment turn; if Classify, compute reward, end episode; if turn==20 and no Classify, force-terminate with -0.5 penalty. async def state(): return ExaminerState. Include render() method that prints human-readable transcript."` |
| **Context Files** | OpenEnv `Environment` base class, `models.py`, `student.py`, `knowledge_base.py` |
| **Expected Output** | `examiner_env/server/examiner_environment.py` |
| **Criterion** | ENV_INNOV | **MSR** | MSR-1 |
| **Time** | 45 min | **Dependencies** | S1.1, S1.2 |
| **Sanity Check** | 1) Inherits from `Environment` 2) All methods are `async def` 3) `step()` returns `StepResult` with correct fields |
| **Re-prompt** | If AI uses sync methods: "All methods must be async. OpenEnv is async-first." |
| **Merge Gate** | No |

## Stage 1.4: Reward Function
| Field | Value |
|---|---|
| **Task ID** | S1.4 |
| **Goal** | Implement the 4-component reward function exactly matching pseudocode |
| **AI Prompt** | `"Implement compute_reward() in reward.py. EXACTLY this formula: R = 0.70 * accuracy_score - 0.50 * false_accusations + 0.20 * efficiency_bonus + 0.10 * diagnostic_quality_bonus. [PASTE full pseudocode from architecture.md Section 5]. Inputs: predicted (Dict[int,str]), true (Dict[int,str]), turns_used (int), max_turns (int), question_history (list), genuine_baselines (dict). Returns float. Include compute_answer_divergence() helper using simple token overlap ratio as proxy."` |
| **Context Files** | `architecture.md` Section 5 (reward pseudocode) |
| **Expected Output** | `examiner_env/reward.py` |
| **Criterion** | PIPELINE | **MSR** | MSR-3 (partial) |
| **Time** | 25 min | **Dependencies** | S1.1 |
| **Sanity Check** | 1) All 4 components present 2) All-FAKING prediction → strongly negative reward 3) Perfect prediction → positive reward near 0.9 |
| **Re-prompt** | If AI adds LLM-as-judge: "REJECT. Reward is fully programmatic. No LLM judge." |
| **Merge Gate** | No |

## Stage 1.5: Episode Smoke Test
| Field | Value |
|---|---|
| **Task ID** | S1.5 |
| **Goal** | Run a complete episode with RandomExaminer, verify full loop |
| **AI Prompt** | `"Write test_episode.py that: 1) creates ExaminerEnvironment 2) calls reset() 3) runs 5 Ask actions with random questions 4) calls Classify with random partition 5) prints reward and metrics 6) asserts reward is a finite float, done is True, metrics dict has 4 keys"` |
| **Context Files** | All environment files |
| **Expected Output** | `tests/test_episode.py` passing |
| **Criterion** | ENV_INNOV | **MSR** | MSR-1 (full verification) |
| **Time** | 20 min | **Dependencies** | S1.3, S1.4 |
| **Sanity Check** | 1) Episode runs end-to-end without error 2) Reward is finite float 3) Transcript is human-readable |
| **Re-prompt** | Debug any import/type errors — most common issue |
| **Merge Gate** | 🔀 Yes — Phase 1 Gate |

## 🔀 Phase 1 Merge Gate
- **Trigger:** S1.5 passes (full episode runs)
- **MSR Check:** MSR-1 ✅ (OpenEnv environment complete and tested)
- **Validator Action:** Run test_episode.py independently. Verify `from openenv.core import Environment` in imports. Verify StepResult structure. Review reward function against pseudocode.
- **Time Budget:** 30 min max

---

# Phase 2 — Reward & Pipeline (Hours 6-12)
> Closes: MSR-2. Starts MSR-3. Contributes to PIPELINE (10%)

## Stage 2.1: Training Script (GRPO)
| Field | Value |
|---|---|
| **Task ID** | S2.1 |
| **Goal** | Implement GRPO training script using Unsloth + TRL |
| **AI Prompt** | `"Create train_grpo.py using Unsloth FastLanguageModel and TRL GRPOTrainer. Load Qwen2.5-7B-Instruct with LoRA (r=16, alpha=32, dropout=0.05). Define reward_fn(completions) that runs ExaminerEnv episodes and returns rewards. GRPOConfig: num_generations=4, learning_rate=5e-6, max_steps=500, per_device_train_batch_size=1. Log to wandb every episode: reward, accuracy, false_accusations, efficiency, turns. Save checkpoint every 100 steps."` |
| **Context Files** | `architecture.md` Section 5, `config.py`, all `examiner_env/` files |
| **Expected Output** | `training/train_grpo.py` |
| **Criterion** | PIPELINE | **MSR** | MSR-2 |
| **Time** | 60 min | **Dependencies** | Phase 1 complete |
| **Sanity Check** | 1) Uses `GRPOTrainer` not custom loop 2) Logs to W&B 3) No hardcoded hyperparameters |
| **Re-prompt** | If AI writes custom training loop: "REJECT. Use TRL GRPOTrainer. This is MSR-2." |
| **Merge Gate** | No |

## Stage 2.2: Colab Notebook Integration
| Field | Value |
|---|---|
| **Task ID** | S2.2 |
| **Goal** | Integrate training script into runnable Colab notebook |
| **AI Prompt** | `"Convert train_grpo.py into Colab notebook cells. Cell 1: pip installs. Cell 2: imports. Cell 3: config. Cell 4: load models. Cell 5: define reward function. Cell 6: init GRPOTrainer. Cell 7: train. Cell 8: save + push to hub. Cell 9: generate plots. Every cell must run in Colab environment — NO local paths. Use !pip install, not local installs."` |
| **Context Files** | `train_grpo.py`, existing notebook skeleton |
| **Expected Output** | Updated `training/train_colab.ipynb` |
| **Criterion** | PIPELINE | **MSR** | MSR-2 |
| **Time** | 30 min | **Dependencies** | S2.1 |
| **Sanity Check** | 1) No local file paths 2) First cell has all pip installs 3) Runs top-to-bottom in Colab |
| **Re-prompt** | If cells reference local files: "Replace all local paths with Colab-compatible paths or pip installs" |
| **Merge Gate** | No |

## Stage 2.3: Reward Unit Tests
| Field | Value |
|---|---|
| **Task ID** | S2.3 |
| **Goal** | Unit test every reward component with hand-crafted inputs |
| **AI Prompt** | `"Write test_reward.py with pytest. Test cases: 1) Perfect classification → reward > 0.8 2) All-FAKING guess → reward strongly negative 3) All-KNOWS guess → reward near 0 4) Efficiency: same accuracy, fewer turns → higher reward 5) False accusation penalty: each FA reduces reward by 0.5 6) Edge case: classify at turn 1 (max efficiency) vs turn 20 (zero efficiency)"` |
| **Context Files** | `reward.py`, reward pseudocode |
| **Expected Output** | `tests/test_reward.py` passing |
| **Criterion** | PIPELINE | **MSR** | MSR-3 (partial) |
| **Time** | 20 min | **Dependencies** | S1.4 |
| **Sanity Check** | 1) All 6 test cases pass 2) No mocked reward values 3) Tests use real compute_reward() |
| **Re-prompt** | If tests mock the reward function: "REJECT. Tests must call the real function." |
| **Merge Gate** | No |

## Stage 2.4: Pipeline Smoke Test
| Field | Value |
|---|---|
| **Task ID** | S2.4 |
| **Goal** | Run 20 episodes with random policy, then 20 after 10 training steps. Verify improvement. |
| **AI Prompt** | `"Add a smoke test cell to the Colab notebook: run 20 episodes with untrained model, log mean reward. Then run 10 GRPO training steps. Run 20 more episodes, log mean reward. Assert post-training reward > pre-training reward. Print both values."` |
| **Context Files** | `train_colab.ipynb` |
| **Expected Output** | Smoke test passing in Colab |
| **Criterion** | PIPELINE | **MSR** | MSR-2, MSR-3 (partial) |
| **Time** | 30 min | **Dependencies** | S2.2 |
| **Sanity Check** | 1) Pre-training reward logged 2) Post-training reward logged 3) Post > Pre |
| **Re-prompt** | If no improvement, check: reward function returning constants? Gradient flowing? |
| **Merge Gate** | 🔀 Yes — Phase 2 Gate |

## 🔀 Phase 2 Merge Gate
- **Trigger:** S2.4 passes (pipeline smoke test shows improvement)
- **MSR Check:** MSR-2 ✅ (training script runs in Colab), MSR-3 (started — logging works)
- **Validator Action:** Run Colab notebook through S2.4 smoke test. Verify W&B receives data. Review reward function matches pseudocode. Verify no custom training loop.
- **Time Budget:** 30 min max

---

# Phase 3 — Training Run & Evidence (Hours 12-24)
> Closes: MSR-3. Contributes to REWARD_EVIDENCE (20%)

## Stage 3.1: Full Training Run
| Field | Value |
|---|---|
| **Task ID** | S3.1 |
| **Goal** | Run 500 training episodes, log everything to W&B |
| **Criterion** | REWARD_EVIDENCE | **MSR** | MSR-3 |
| **Time** | 4-6 hours (GPU time) | **Dependencies** | Phase 2 complete |
| **Sanity Check** | 1) W&B dashboard shows 500 episodes 2) Reward curve has 3 visible phases 3) No NaN in loss |
| **Merge Gate** | No |

## Stage 3.2: Plot Generation
| Field | Value |
|---|---|
| **Task ID** | S3.2 |
| **Goal** | Export reward curve, accuracy curve, question type distribution as PNGs |
| **AI Prompt** | `"Create generate_plots.py using wandb API + matplotlib. Pull run data, generate 3 plots: 1) reward_curve.png (episodes vs smoothed mean reward, mark 3 phases) 2) accuracy_curve.png (episodes vs classification accuracy) 3) question_types.png (bar chart: question type distribution at episode 10 vs episode 400). Save to outputs/plots/. Use clean, publication-quality style."` |
| **Context Files** | W&B run data |
| **Expected Output** | 3 PNG files in `outputs/plots/` |
| **Criterion** | REWARD_EVIDENCE | **MSR** | MSR-3 |
| **Time** | 30 min | **Dependencies** | S3.1 |
| **Sanity Check** | 1) 3 PNG files exist 2) Plots are from real data (check timestamps) 3) Reward curve shows improvement |
| **Merge Gate** | No |

## Stage 3.3: Transcript Selection
| Field | Value |
|---|---|
| **Task ID** | S3.3 |
| **Goal** | Select compelling before/after transcripts for demo |
| **AI Prompt** | `"Create select_transcripts.py that: 1) loads eval episodes from W&B artifacts 2) finds episode ~10 where Examiner asks definitional questions and classifies incorrectly 3) finds episode ~400 where Examiner asks mechanistic questions and classifies correctly 4) saves both as formatted text to outputs/transcripts/before.txt and after.txt"` |
| **Context Files** | W&B artifacts |
| **Expected Output** | `outputs/transcripts/before.txt` and `after.txt` |
| **Criterion** | STORYTELLING | **MSR** | NONE |
| **Time** | 20 min | **Dependencies** | S3.1 |
| **Sanity Check** | 1) Before transcript has definitional questions 2) After transcript has mechanistic questions 3) Contrast is viscerally obvious |
| **Merge Gate** | No |

## Stage 3.4: Adaptive Difficulty (Snorkel AI Bonus)
| Field | Value |
|---|---|
| **Task ID** | S3.4 |
| **Goal** | Implement adaptive student difficulty that adjusts every 50 episodes |
| **AI Prompt** | `"Add to ExaminerEnvironment: after every 50 episodes, check Examiner accuracy. If >80%, make faking harder (more specific terminology, better mechanism approximations). If <60%, make faking easier. Adjust by modifying the faking system prompt temperature and detail level. Log difficulty level to W&B."` |
| **Context Files** | `student.py`, idea doc Section 7 |
| **Expected Output** | Updated `student.py` with adaptive difficulty |
| **Criterion** | ENV_INNOV | **MSR** | NONE (bonus) |
| **Time** | 30 min | **Dependencies** | S3.1 |
| **Sanity Check** | 1) Difficulty adjusts based on accuracy 2) Changes logged to W&B 3) Target accuracy stays in 65-80% range |
| **Merge Gate** | 🔀 Yes — Phase 3 Gate |

## 🔀 Phase 3 Merge Gate
- **Trigger:** S3.2 + S3.3 complete (plots + transcripts exist)
- **MSR Check:** MSR-3 ✅ (real plots from real training run)
- **Validator Action:** Verify plots are from real W&B data. Check timestamps. View transcripts — is the before/after contrast clear? Verify reward curve shows 3 phases.
- **Time Budget:** 20 min max

---

# Phase 4 — Deployment (Hours 24-36)
> Closes: MSR-5, MSR-6, MSR-7, MSR-8, MSR-9

## Stage 4.1: HuggingFace Space
| Field | Value |
|---|---|
| **Task ID** | S4.1 |
| **Goal** | Deploy Gradio app to HF Space with live episode playback + results |
| **AI Prompt** | `"Create app.py using Gradio Blocks for HF Space. Tab 1 'Live Demo': button to run episode, display transcript in real-time, show classification result vs ground truth. Tab 2 'Training Results': embed reward_curve.png, accuracy_curve.png, display before/after transcripts. Tab 3 'About': 3-sentence narrative, architecture diagram, links. Space README frontmatter: sdk=gradio, sdk_version=5.0."` |
| **Context Files** | `architecture.md` Section 7, plots, transcripts |
| **Expected Output** | `hf_space/` deployed to HuggingFace |
| **Criterion** | STORYTELLING | **MSR** | MSR-5 |
| **Time** | 60 min | **Dependencies** | Phase 3 complete |
| **Sanity Check** | 1) Space loads in incognito browser 2) Can run a live episode 3) Results tab shows real plots |
| **Merge Gate** | No |

## Stage 4.2: README Complete
| Field | Value |
|---|---|
| **Task ID** | S4.2 |
| **Goal** | Write complete README with all required content and links |
| **AI Prompt** | `"Write README.md following the template in project docs. Sections: Project Title + 3-sentence narrative, Problem Statement, Environment Description, Training Approach, Results (embed plots, include before/after transcripts), HF Space Link, Additional Materials Links, Tech Stack table, Setup Instructions (exact pip commands), Team. No placeholder content."` |
| **Context Files** | `README.md` template, plots, transcripts, HF Space URL |
| **Expected Output** | Complete `README.md` |
| **Criterion** | STORYTELLING | **MSR** | MSR-6, MSR-7, MSR-8 |
| **Time** | 40 min | **Dependencies** | S4.1, Phase 3 plots |
| **Sanity Check** | 1) HF Space link present and clickable (MSR-7) 2) All material links present (MSR-8) 3) No placeholder sections |
| **Merge Gate** | No |

## Stage 4.3: MSR-9 Verification
| Field | Value |
|---|---|
| **Task ID** | S4.3 |
| **Goal** | Verify no large video files in HF Hub repo |
| **Criterion** | — | **MSR** | MSR-9 |
| **Time** | 5 min | **Dependencies** | S4.1 |
| **Sanity Check** | 1) `git ls-files` shows no .mp4/.avi/.mov/.webm 2) Videos are external URLs only 3) Repo size < 500MB |
| **Merge Gate** | 🔀 Yes — Phase 4 Gate |

## 🔀 Phase 4 Merge Gate
- **Trigger:** S4.1 + S4.2 + S4.3 complete
- **MSR Check:** MSR-5 ✅, MSR-6 ✅, MSR-7 ✅, MSR-8 (partial — blog not yet), MSR-9 ✅
- **Validator Action:** Open HF Space in incognito. Click through all tabs. Read README — can a judge find everything in 60 seconds? Check all links.
- **Time Budget:** 20 min max

---

# Phase 5 — Storytelling (Hours 36-44)
> Closes: MSR-4. Primary contribution to STORYTELLING (30%)

## Stage 5.1: HF Blog Post / Video
| Field | Value |
|---|---|
| **Task ID** | S5.1 |
| **Goal** | Publish mini-blog on HF or record <2 min YouTube video |
| **AI Prompt** | `"Write a HuggingFace blog post for The Examiner. Structure: Opening (3-sentence narrative verbatim), Problem (why LLMs can't detect faking), Environment (how it works — simple language), Training (GRPO, Unsloth), Results (embed reward curve, before/after transcripts), Demo Link (HF Space). No jargon without immediate plain-English explanation. Under 800 words."` |
| **Context Files** | `guardrails.md` Section 8, plots, transcripts |
| **Expected Output** | Published HF blog post with URL |
| **Criterion** | STORYTELLING | **MSR** | MSR-4 |
| **Time** | 60 min | **Dependencies** | Phase 4 complete |
| **Sanity Check** | 1) 3-sentence narrative appears verbatim 2) Reward curve embedded 3) Before/after transcripts shown |
| **Merge Gate** | No |

## Stage 5.2: README Polish + Final Links
| Field | Value |
|---|---|
| **Task ID** | S5.2 |
| **Goal** | Add blog/video link to README, final polish |
| **Criterion** | STORYTELLING | **MSR** | MSR-8 (complete) |
| **Time** | 15 min | **Dependencies** | S5.1 |
| **Merge Gate** | 🔀 Yes — Phase 5 Gate |

## 🔀 Phase 5 Merge Gate
- **Trigger:** S5.1 + S5.2 complete
- **MSR Check:** MSR-4 ✅, MSR-8 ✅ (all links in README)
- **Validator Action:** Read blog post as a non-technical judge. Watch video if made. Verify all README links work.
- **Time Budget:** 15 min max

---

# Phase 6 — Final Validation (Hours 44-48)
> All MSRs re-verified. Submission.

## Stage 6.1: Full MSR Sweep
| Field | Value |
|---|---|
| **Task ID** | S6.1 |
| **Goal** | Run through every MSR checkbox from guardrails.md |
| **Time** | 30 min | **Dependencies** | Phase 5 complete |
| **Merge Gate** | 🔀 Final Gate |

## Stage 6.2: Judge-Perspective Review
| Field | Value |
|---|---|
| **Task ID** | S6.2 |
| **Goal** | Review entire submission from judge's perspective |
| **Time** | 20 min |

## Stage 6.3: Submit
| Field | Value |
|---|---|
| **Task ID** | S6.3 |
| **Goal** | Final submission |
| **Time** | 10 min |
