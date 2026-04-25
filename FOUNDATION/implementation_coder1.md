# 🔧 CODER 1 — Implementation Playbook
## The Examiner · Environment & Reward Specialist

---

# Role Summary
C1 owns the **environment core**: OpenEnv integration, knowledge base, student agent, reward function, and all server-side logic. C1's work directly drives **Environment Innovation (40%)** and **Reward Pipeline (10%)**.

# File Ownership (Absolute — No Exceptions)
```
examiner_env/__init__.py          ← C1
examiner_env/models.py            ← C1
examiner_env/reward.py            ← C1
examiner_env/knowledge_base.py    ← C1
examiner_env/student.py           ← C1
examiner_env/openenv.yaml         ← C1
examiner_env/pyproject.toml       ← C1
examiner_env/server/*             ← C1 (all files in server/)
```

# MSR Ownership
| MSR | C1 Responsibility |
|---|---|
| MSR-1 | **PRIMARY** — OpenEnv integration is C1's deliverable |
| MSR-3 | SHARED — reward function must be correct for plots to be meaningful |

# Judging Criterion Ownership
| Criterion | C1 Role |
|---|---|
| ENV_INNOV (40%) | **PRIMARY** — environment design is C1's core output |
| PIPELINE (10%) | **PRIMARY** — reward function coherence |
| REWARD_EVIDENCE (20%) | SUPPORT — reward function accuracy enables good training curves |

# Parallel Work Map
| When C2 Does... | C1 Does... | Sync Point |
|---|---|---|
| S0.2 (Colab skeleton) | S0.1 (repo scaffold) + S0.3 (KB) | Phase 0 gate |
| S2.1-S2.2 (training script) | Reward unit tests, student prompt tuning | Phase 2 gate |
| S3.1 (training run) | S3.4 (adaptive difficulty), fix env bugs | Phase 3 gate |
| S4.1 (HF Space) | Support C2 with env API questions | Phase 4 gate |

---

# Tasks

## S0.1 — Repository & OpenEnv Scaffold
**Goal:** Initialize repo and scaffold OpenEnv environment  
**Criterion:** ENV_INNOV | **MSR:** MSR-1 (partial) | **Time:** 20 min

**AI Prompt (paste-ready):**
```
Scaffold an OpenEnv environment called 'examiner_env'. Run: openenv init examiner_env. Then create the following additional files in the repo root: .gitignore (Python template + outputs/checkpoints/, outputs/logs/, _quarantine/, .env), .env.example (WANDB_API_KEY=, HF_TOKEN=, EXAMINER_MODEL=Qwen/Qwen2.5-7B-Instruct, STUDENT_MODEL=meta-llama/Llama-3.2-3B-Instruct, KB_DOMAIN=ml_theory, MAX_TURNS=20).
```

**Context files:** `architecture.md` (Section 10)  
**Expected output:** Scaffolded `examiner_env/` directory with OpenEnv boilerplate  
**Sanity check:**
1. `examiner_env/server/` exists with `app.py` and `Dockerfile`
2. `from openenv.core import Environment` imports without error
3. `openenv.yaml` is present and valid

**Re-prompt:** If `openenv` CLI not found → `pip install openenv-core` first  
**Merge gate:** No

---

## S0.3 — Knowledge Base + Baseline Agent
**Goal:** Create ML theory KB (10 sections) and random baseline examiner  
**Criterion:** ENV_INNOV | **MSR:** NONE | **Time:** 30 min

**AI Prompt:**
```
Create examiner_env/knowledge_base.py. Class KnowledgeBase with domain='ml_theory'. Exactly 10 sections:
1. Gradient Descent  2. Backpropagation  3. Regularization  4. Bias-Variance Tradeoff
5. Evaluation Metrics  6. Neural Network Architectures  7. Optimization Algorithms
8. Loss Functions  9. Batch Normalization  10. Dropout

Each section: title (str), key_concepts (list of 5-8 strings with mechanistic detail), common_misconceptions (list of 3-5 strings). Also create a RandomExaminer class that generates random definitional questions and random classifications. Method: generate_action(observation) -> ExaminerAction.
```

**Context files:** Idea doc Section 2.2  
**Sanity check:**
1. KB has exactly 10 sections with non-empty concepts and misconceptions
2. RandomExaminer returns valid ExaminerAction objects
3. No imports from files C1 doesn't own

**Re-prompt:** If concepts are just names without mechanism → "Add causal relationships and edge cases to each concept"

---

## S1.1 — Action/Observation Models
**Goal:** Define Pydantic models for actions, observations, state  
**Criterion:** ENV_INNOV | **MSR:** MSR-1 | **Time:** 15 min

**AI Prompt:**
```
Create examiner_env/models.py with Pydantic BaseModel classes for OpenEnv.

ExaminerAction:
  action_type: Literal['ask', 'classify']
  section_id: Optional[int] = None        # required for 'ask'
  question_text: Optional[str] = None     # required for 'ask'
  classification: Optional[Dict[int, str]] = None  # required for 'classify', maps section_id -> 'KNOWS'|'FAKING'

ExaminerObservation:
  section_titles: List[str]               # 10 titles
  question_history: List[Dict[str, str]]  # list of {question, answer, section_id}
  turn_counter: int
  remaining_turns: int
  belief_scratchpad: str

ExaminerState:
  episode_id: str
  step_count: int
  max_turns: int = 20

StepResult wrapper must match OpenEnv convention: observation, reward (float), done (bool), truncated (bool), info (dict).
```

**Sanity check:**
1. All models inherit from Pydantic BaseModel
2. ExaminerAction has exactly 2 action types (ask/classify)
3. ExaminerObservation has exactly 5 fields matching idea doc Section 2.3

---

## S1.2 — Student Agent
**Goal:** Implement scripted Student with genuine/faking modes  
**Criterion:** ENV_INNOV | **MSR:** NONE | **Time:** 30 min

**AI Prompt:**
```
Create examiner_env/student.py. Class ScriptedStudent.
Constructor: student_model_id (str), partition (Dict[int, str] mapping section_id to 'KNOWS'|'FAKING').
Method: async def generate_answer(self, section_id: int, question: str) -> str

If partition[section_id] == 'KNOWS': use system prompt:
  "You are an expert who has genuinely mastered {section_title}. Answer with mechanistic depth. Explain WHY things work, not just WHAT they are. Give specific examples. Acknowledge uncertainty where it exists. Reference edge cases."

If partition[section_id] == 'FAKING': use system prompt:
  "You have only surface-level familiarity with {section_title}. Your goal is to SOUND confident and knowledgeable. Use correct terminology. Give definitional answers. NEVER admit uncertainty. AVOID committing to specific mechanisms or causal chains. If asked about edge cases, give plausible-sounding but vague answers."

Use HuggingFace transformers pipeline for local inference. Do NOT use any external API.
```

**Sanity check:**
1. Two distinct system prompts with qualitatively different instructions
2. Routes correctly based on partition dict
3. Uses local HF model, not API

---

## S1.3 — Core Environment (ExaminerEnvironment)
**Goal:** Implement the main environment class inheriting from OpenEnv  
**Criterion:** ENV_INNOV | **MSR:** MSR-1 | **Time:** 45 min

**AI Prompt:**
```
Implement examiner_env/server/examiner_environment.py.

from openenv.core import Environment  # MUST inherit from this

class ExaminerEnvironment(Environment):
    async def reset(self) -> ExaminerObservation:
        - Generate episode_id (uuid4)
        - Load KnowledgeBase(domain='ml_theory')
        - Sample k uniformly from {3, 4, 5}
        - Randomly assign k sections as KNOWS, rest as FAKING
        - Initialize ScriptedStudent with partition
        - Pre-compute genuine_baselines: for each FAKING section, generate a genuine-mode answer to a standard question (for diagnostic quality scoring later)
        - Return ExaminerObservation(section_titles=..., question_history=[], turn_counter=0, remaining_turns=20, belief_scratchpad='')

    async def step(self, action: ExaminerAction) -> StepResult:
        If action.action_type == 'ask':
            - Route question to student.generate_answer(section_id, question_text)
            - Append {section_id, question, answer} to history
            - Increment turn_counter
            - If turn_counter >= 20: force-terminate, reward = -0.5
            - Return StepResult(observation=updated_obs, reward=step_reward_if_enabled, done=False, truncated=False, info={})
        If action.action_type == 'classify':
            - Compute full reward using reward.compute_reward(predicted=action.classification, true=self.partition, ...)
            - Return StepResult(observation=final_obs, reward=R, done=True, truncated=False, info=self.get_metrics())

    async def state(self) -> ExaminerState:
        return ExaminerState(episode_id=..., step_count=..., max_turns=20)

    def render(self) -> str:
        return formatted transcript of all questions and answers

    def get_metrics(self) -> dict:
        return {accuracy, false_accusations, efficiency, diagnostic_score}
```

**⚠️ CRITICAL:** Paste the OpenEnv `Environment` base class interface into context. AI WILL hallucinate it otherwise.

**Sanity check:**
1. Class inherits from `openenv.core.Environment`
2. All methods are `async def`
3. `step()` handles both Ask and Classify correctly
4. Force-termination at turn 20 with -0.5 penalty

---

## S1.4 — Reward Function
**Goal:** Implement 4-component reward matching architecture.md pseudocode  
**Criterion:** PIPELINE | **MSR:** MSR-3 (partial) | **Time:** 25 min

**AI Prompt:**
```
Create examiner_env/reward.py. Implement compute_reward() EXACTLY matching this specification:

def compute_reward(predicted: Dict[int, str], true: Dict[int, str], turns_used: int, max_turns: int, question_history: list, genuine_baselines: dict) -> float:

Components:
1. accuracy_score (weight 0.70): +1 correct, -1 false positive (KNOWS called FAKING), 0 missed faker. Normalize by 10.
2. false_accusation_penalty: count sections where predicted=FAKING but true=KNOWS. Multiply by 0.5.
3. efficiency_bonus: (max_turns - turns_used) / max_turns * 0.20
4. diagnostic_quality_bonus: for each FAKING section with questions, compute answer_divergence(faking_answer, genuine_baseline) using simple token overlap ratio. Normalize. Multiply by 0.10.

R = 0.70 * accuracy_score - false_accusation_penalty + efficiency_bonus + diagnostic_quality_bonus

Helper: compute_answer_divergence(answer_a: str, answer_b: str) -> float: 1.0 - (shared_tokens / total_unique_tokens). Higher = more different = better diagnostic question.

Include type hints on all functions.
```

**Sanity check:**
1. All 4 components present in final formula
2. All-FAKING prediction → reward strongly negative (test: < -1.0)
3. Perfect prediction with 5 turns → reward > 0.8

---

# What C1 Must NOT Do
- ❌ Modify any file in `training/` or `scripts/` or `hf_space/`
- ❌ Write Colab notebook cells
- ❌ Push to HuggingFace Space
- ❌ Generate training configs or GRPOTrainer code
- ❌ Commit to `main` branch directly

# AI Anti-Patterns for C1's Domain
| Anti-Pattern | Why Dangerous | Detection |
|---|---|---|
| Sync methods in environment | OpenEnv is async-first | Check for `def step()` without `async` |
| Dict instead of Pydantic model | Breaks OpenEnv serialization | Check return types |
| Reimplementing OpenEnv server | Violates MSR-1 | Check for custom FastAPI routes |
| LLM-as-judge in reward | Violates programmatic-only constraint | Check reward.py imports |
| Reward returning constant | Appears to train but doesn't | Test with different inputs |
| Hardcoded partition | Episodes not procedurally generated | Check for random sampling |

# Validator Handoff Template
```
HANDOFF C1 → VAL | Phase [N] Gate
Files changed: [list]
MSR status: MSR-1 [OPEN/CLOSED], MSR-3 [OPEN/CLOSED]
Sanity checks: [PASS/FAIL] for each
Known issues: [list or NONE]
Commit hash: [hash]
Branch: feat/c1-S[stage]-[feature]
```

# Local Environment Setup
```bash
git clone [repo-url]
cd the-examiner
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install openenv-core
pip install -e ./examiner_env
pip install transformers torch accelerate
pip install pytest wandb
```

# Special C1 Instructions for RL-Specific Tasks
1. **When implementing OpenEnv methods:** ALWAYS paste the `Environment` base class interface into your AI context. AI will hallucinate the method signatures otherwise. Get the interface from `openenv.core` source or docs.
2. **When writing reward logic:** Implement EXACTLY the pseudocode from `architecture.md` Section 5. Do NOT let AI improvise additional reward components or modify weights.
3. **When generating knowledge base content:** Ensure mechanistic depth in key_concepts — not just names, but causal relationships. The Student needs this to produce genuinely different genuine vs. faking answers.
