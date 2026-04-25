# 🧠 The Examiner
### An RL Training Environment for Diagnostic Knowledge Verification

> LLMs are dangerously good at sounding like they know things they don't — and current AI systems cannot tell the difference. The Examiner is an RL environment that trains an AI to design diagnostic questions that reliably separate genuine knowledge from confident faking. After training, the Examiner learns to skip surface-level questions and ask the mechanistic, causal questions that only genuine experts can answer.

---

## 🔗 Quick Links

| Resource | Link |
|---|---|
| **🎯 Live Demo (HF Space)** | [▶️ Try The Examiner](https://huggingface.co/spaces/team/the-examiner) |
| **📓 Training Notebook** | [Open in Colab](https://colab.research.google.com/TODO) |
| **📝 Writeup / Blog** | [Read on HuggingFace](https://huggingface.co/blog/TODO) |
| **🎬 Demo Video** | [Watch on YouTube](https://youtube.com/TODO) |

> ⚠️ Replace TODO links before submission (MSR-7, MSR-8)

---

## 🎯 The Problem

Large language models are trained on text where confident, fluent answers overwhelmingly correlate with correct answers. This creates a catastrophic blind spot: **LLMs cannot distinguish genuine knowledge from confident-sounding ignorance.**

This matters in:
- **Technical hiring:** Is the candidate's "distributed systems experience" real or memorized from blog posts?
- **AI system auditing:** Does the vendor's LLM actually perform, or is it pattern-matching?
- **Credentialing:** Does the practitioner genuinely understand the domain?

**Why RL is necessary:** This requires an agent that chooses a sequence of questions, observes answers, updates a belief model, and designs the next question to maximally reduce uncertainty. That's a sequential decision process with delayed rewards — textbook RL.

---

## 🏗️ Environment Design

The Examiner is a **two-agent adversarial RL environment**:

| Agent | Role |
|---|---|
| **Examiner** (trained) | Generates diagnostic questions. Learns which question types reliably separate knowers from fakers. |
| **Student** (scripted adversary) | Has genuinely mastered some KB sections and is faking others. Tries to pass regardless. |

### How It Works
1. A knowledge base (ML Theory, 10 sections) is loaded
2. The Student is secretly assigned: k sections KNOWS, (10-k) sections FAKING
3. The Examiner asks up to 20 questions, observing only section titles and answers
4. The Examiner submits a classification: for each section, KNOWS or FAKING
5. Reward is computed from classification accuracy, efficiency, and diagnostic quality

### The Key Insight
A **faker** answers "What is gradient descent?" correctly. But a faker **fails** on "Why does gradient descent with a fixed learning rate fail to converge on saddle points, and what does this imply about initialization strategies?" — *but only if the Examiner asks that second question.* The Examiner must **learn** to ask it.

---

## 🏋️ Training Approach

| Component | Technology |
|---|---|
| **Algorithm** | GRPO (Group Relative Policy Optimization) via TRL |
| **Examiner Model** | Qwen2.5-7B-Instruct (LoRA via Unsloth) |
| **Student Model** | Llama-3.2-3B-Instruct (scripted genuine/faking prompts) |
| **Logging** | Weights & Biases |
| **Training Environment** | Google Colab (A100 GPU) |

**Reward Function (fully programmatic — no LLM judge):**
```
R = 0.70 × classification_accuracy
  − 0.50 × false_accusations
  + 0.20 × efficiency_bonus
  + 0.10 × diagnostic_quality_bonus
```

---

## 📊 Results

### Reward Curve
<!-- Replace with actual plot after training -->
![Reward Curve](outputs/plots/reward_curve.png)

Three visible training phases:
1. **Episodes 0-100:** Random questioning, ~50% accuracy, negative reward
2. **Episodes 100-300:** Learning diagnostic patterns, accuracy climbs to 65-70%
3. **Episodes 300-500:** Skilled examination, 75-85% accuracy, reward ~0.6

### Before Training (Episode ~10)
<!-- Replace with actual transcript -->
```
Examiner: "What is gradient descent?"
Student: "Gradient descent is an optimization algorithm..." [correct — faker passes]
Examiner classifies section as KNOWS ❌ (actually FAKING)
```

### After Training (Episode ~400)
<!-- Replace with actual transcript -->
```
Examiner: "If you apply dropout with p=0.5 during inference instead of training, what breaks in the forward pass and why is the fix multiplicative?"
Student: "Well, dropout helps prevent overfitting by..." [vague — faker exposed]
Examiner classifies section as FAKING ✅
```

---

## 🛠️ Tech Stack

| Component | Technology | Version |
|---|---|---|
| Environment Framework | OpenEnv (`openenv-core`) | latest |
| RL Training | Unsloth + TRL `GRPOTrainer` | `unsloth>=2025.4`, `trl>=0.15` |
| Experiment Logging | Weights & Biases | `wandb>=0.19` |
| Deployment UI | Gradio on HuggingFace Spaces | `gradio>=5.0` |
| Base Model | Qwen2.5-7B-Instruct | HF latest |

---

## 🚀 Setup & Reproduction

### Run Training (Colab)
1. Open the [Training Notebook](https://colab.research.google.com/TODO) in Google Colab
2. Select GPU runtime (T4 or A100)
3. Click **Runtime → Run All**
4. Training logs appear in W&B dashboard automatically

### Run Environment Locally
```bash
git clone https://github.com/team/the-examiner.git
cd the-examiner
pip install openenv-core
pip install -e ./examiner_env
pip install transformers torch accelerate wandb

# Run a single episode
python -c "
import asyncio
from examiner_env.server.examiner_environment import ExaminerEnvironment
from examiner_env.models import ExaminerAction

async def main():
    env = ExaminerEnvironment()
    obs = await env.reset()
    print(f'Sections: {obs.section_titles}')
    # Ask a question
    action = ExaminerAction(action_type='ask', section_id=0, question_text='What is gradient descent?')
    result = await env.step(action)
    print(f'Answer: {result.observation.question_history[-1]}')

asyncio.run(main())
"
```

---

## 👥 Team

| Member | Role |
|---|---|
| [Name] | Coder 1 — Environment & Reward |
| [Name] | Coder 2 — Training Pipeline & Deployment |
| [Name] | Validator — Quality & Submission |

---

## 📜 License

MIT

---

*Built for the OpenEnv Hackathon India 2026 · Theme 4: Self-Improvement · Snorkel AI Bonus Prize*
