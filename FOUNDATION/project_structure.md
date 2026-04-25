# 📁 PROJECT STRUCTURE — THE EXAMINER
## Source of Truth for File Ownership & AI Scaffolding · Locked After Phase 0

---

```
the-examiner/
│
├── README.md                                    SHARED (VAL finalizes) | MSR-6,7,8 | STORYTELLING
├── guardrails.md                                SHARED | All MSRs | ALL CRITERIA
├── architecture.md                              SHARED | Reference
├── implementation_plan.md                       SHARED | Reference
├── implementation_coder1.md                     C1 | Reference
├── implementation_coder2.md                     C2 | Reference
├── implementation_validator.md                  VAL | Reference
├── merge_procedure.md                           SHARED | Reference
├── mistakes.md                                  SHARED | Living log
├── context_primer.md                            VAL | AI primer
├── submission_checklist.md                      VAL | Final validation
├── project_structure.md                         SHARED | This file
├── .gitignore                                   SHARED
├── .env.example                                 SHARED
│
├── examiner_env/                                C1 OWNS (except client.py)
│   ├── __init__.py                              C1 | MSR-1 | Exports main classes
│   ├── models.py                                C1 | MSR-1 | Pydantic: ExaminerAction, ExaminerObservation, ExaminerState
│   ├── reward.py                                C1 | PIPELINE | 4-component reward function
│   ├── knowledge_base.py                        C1 | ENV_INNOV | ML theory KB (10 sections)
│   ├── student.py                               C1 | ENV_INNOV | ScriptedStudent (genuine/faking modes)
│   ├── client.py                                C2 | MSR-1 | EnvClient subclass for remote access
│   ├── openenv.yaml                             C1 | MSR-1 | Environment manifest
│   ├── pyproject.toml                           C1 | Package config + dependencies
│   └── server/
│       ├── examiner_environment.py              C1 | MSR-1 | ExaminerEnvironment(Environment) — core env logic
│       ├── app.py                               C1 | MSR-1 | FastAPI server entry point
│       ├── requirements.txt                     C1 | Docker dependencies
│       └── Dockerfile                           C1 | Container definition
│
├── training/                                    C2 OWNS
│   ├── config.py                                C2 | MSR-2 | TrainingConfig dataclass (all hyperparams)
│   ├── train_grpo.py                            C2 | MSR-2 | GRPO training script (Unsloth + TRL)
│   ├── train_colab.ipynb                        C2 | MSR-2 | Runnable Colab notebook (top-to-bottom)
│   └── eval.py                                  C2 | MSR-3 | Evaluation runner + metric collection
│
├── scripts/                                     C2 OWNS
│   ├── generate_plots.py                        C2 | MSR-3 | W&B data → matplotlib PNGs
│   ├── select_transcripts.py                    C2 | STORYTELLING | Pick before/after transcripts
│   └── push_to_hub.py                           C2 | MSR-5 | HF Hub upload utility
│
├── hf_space/                                    C2 OWNS
│   ├── app.py                                   C2 | MSR-5 | Gradio Blocks (3 tabs)
│   ├── requirements.txt                         C2 | MSR-5 | Space dependencies
│   ├── README.md                                C2 | MSR-5 | Space card (sdk: gradio)
│   └── assets/                                  C2 | STORYTELLING
│       ├── reward_curve.png                     C2 | MSR-3 | From generate_plots.py
│       ├── accuracy_curve.png                   C2 | MSR-3 | From generate_plots.py
│       └── architecture.png                     VAL | STORYTELLING | System diagram
│
├── outputs/                                     GENERATED
│   ├── plots/                                   C2 | MSR-3 | Committed to repo
│   │   ├── reward_curve.png                     — | Smoothed reward vs episodes
│   │   ├── accuracy_curve.png                   — | Classification accuracy vs episodes
│   │   └── question_types.png                   — | Question type distribution shift
│   ├── transcripts/                             C2 | STORYTELLING | Committed
│   │   ├── before.txt                           — | Episode ~10 transcript
│   │   └── after.txt                            — | Episode ~400 transcript
│   ├── checkpoints/                             — | GITIGNORED (pushed to HF Hub)
│   └── logs/                                    — | GITIGNORED (in W&B)
│
├── tests/                                       VAL OWNS
│   ├── test_reward.py                           VAL | PIPELINE | 6 reward function test cases
│   ├── test_environment.py                      VAL | MSR-1 | OpenEnv compliance tests
│   └── test_episode.py                          VAL | ENV_INNOV | Full episode smoke test
│
└── _quarantine/                                 — | Never committed | Out-of-scope AI output
```

---

## Ownership Summary

| Owner | Files | Primary MSRs | Primary Criteria |
|---|---|---|---|
| **C1** | `examiner_env/` (except `client.py`) | MSR-1 | ENV_INNOV (40%), PIPELINE (10%) |
| **C2** | `training/`, `scripts/`, `hf_space/`, `examiner_env/client.py`, `outputs/` | MSR-2, MSR-3, MSR-5, MSR-9 | REWARD_EVIDENCE (20%), PIPELINE (10%) |
| **VAL** | `tests/`, `submission_checklist.md`, `context_primer.md` | All (verification) | All (quality gate) |
| **SHARED** | `README.md`, `guardrails.md`, `mistakes.md`, docs | MSR-6,7,8 | STORYTELLING (30%) |

## .gitignore Contents
```
.env
outputs/checkpoints/
outputs/logs/
_quarantine/
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.ipynb_checkpoints/
```
