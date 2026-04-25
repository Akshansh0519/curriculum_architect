# \[cite\_start]THE EXAMINER \[cite: 1]

## \[cite\_start]An RL Training Environment for Diagnostic Knowledge Verification \[cite: 2]

\[cite\_start]**Full Project Problem Statement · OpenEnv Hackathon India 2026** \[cite: 3]

|Category|Details|
|-|-|
|**Primary Theme**|\[cite\_start]Theme 4 — Self-Improvement (Self-Play, Adaptive Curricula) \[cite: 4]|
|**Secondary Theme**|\[cite\_start]Theme 1 — Multi-Agent Interactions (Adversarial, Theory-of-Mind) \[cite: 4]|
|**Bonus Prize Target**|\[cite\_start]Snorkel AI — Simulated Experts-in-the-Loop \[cite: 4]|
|**Wild Card Theme**|\[cite\_start]Theme 5 — Wild Card (novel capability; impress us) \[cite: 4]|
|**Reward Signal**|\[cite\_start]Fully programmatic — no LLM judge, no human rater, no entailment model \[cite: 4]|

## \[cite\_start]1. The Problem — What LLMs Cannot Do \[cite: 5]

\[cite\_start]Large language models are trained on text where confident, fluent answers overwhelmingly correlate with correct answers. \[cite: 6] \[cite\_start]This creates a catastrophic blind spot: LLMs cannot distinguish genuine knowledge from confident-sounding ignorance. \[cite: 7]

\[cite\_start]Give an LLM a candidate who says "I've worked extensively with distributed consensus protocols" and asks it to determine whether this person truly understands the subject — it will say yes, because the sentence is grammatically confident and semantically plausible. \[cite: 8] \[cite\_start]This is not a knowledge problem. \[cite: 9] \[cite\_start]It is a diagnostic reasoning problem. \[cite: 9] \[cite\_start]The LLM has no training signal for the following cognitive task: \[cite: 10]

> \[cite\_start]\*\*Core Cognitive Gap:\*\* Design a sequence of questions whose answers are fundamentally different depending on whether the respondent genuinely understands the subject vs. is producing fluent-sounding approximations. \[cite: 11] \[cite\_start]Then use those differences to make a reliable classification. \[cite: 11]

\[cite\_start]This capability matters in three high-stakes real-world workflows that currently operate without it: \[cite: 12]

* \[cite\_start]**Technical hiring:** Interviewers must determine whether a candidate truly knows distributed systems, or has read enough Medium posts to sound credible. \[cite: 13] \[cite\_start]Getting it wrong costs 6 months of salary and a failed project. \[cite: 14]
* \[cite\_start]**AI system auditing:** Regulators and enterprise buyers must determine whether a vendor's LLM actually performs the claimed capability, or is pattern-matching to produce plausible-looking outputs. \[cite: 15] \[cite\_start]Getting it wrong means deploying a system that fails silently in production. \[cite: 16]
* \[cite\_start]**Medical and legal credentialing:** Boards must determine whether a practitioner genuinely understands a domain or has memorized surface patterns. \[cite: 17] \[cite\_start]Getting it wrong has life-or-death consequences. \[cite: 18]

> \[cite\_start]\*\*Why RL is necessary:\*\* This cannot be solved with RAG or prompting. \[cite: 19] \[cite\_start]It requires an agent that chooses a sequence of questions, observes answers, updates a belief model about what the respondent truly knows, and then designs the next question to maximally reduce remaining uncertainty. \[cite: 19] \[cite\_start]That is a sequential decision process with delayed rewards — textbook RL territory. \[cite: 19]

## \[cite\_start]2. Environment Specification \[cite: 20]

### \[cite\_start]2.1 High-Level Architecture \[cite: 21]

\[cite\_start]The Examiner is a two-agent adversarial RL environment with the following structure: \[cite: 22]

|Agent|Role|Description|
|-|-|-|
|**Examiner**|Trained RL agent|\[cite\_start]Generates diagnostic questions. \[cite: 23] \[cite\_start]Learns which question types reliably separate genuine knowers from fakers. \[cite: 23] \[cite\_start]This is the model you train with Unsloth/TRL. \[cite: 23]|
|**Student**|Adversarial counterpart|\[cite\_start]Has genuinely mastered some sections of a knowledge base and is faking familiarity with others. \[cite: 23] \[cite\_start]Attempts to pass the examination regardless of true knowledge. \[cite: 23] \[cite\_start]Can be a scripted adversary (v1) or a second RL-trained model (v2). \[cite: 23]|
|**Environment**|Episode manager|\[cite\_start]Maintains the knowledge base partition (what the student truly knows vs. fakes), runs the examination loop, computes rewards, and generates new episodes procedurally. \[cite: 23]|

### \[cite\_start]2.2 The Knowledge Base \[cite: 24]

\[cite\_start]Each episode uses a knowledge base (KB) consisting of a technical document or structured fact set partitioned into exactly 10 sections. \[cite: 25]

\[cite\_start]Candidate domains (pick one for the hackathon, keep it narrow): \[cite: 26]

* \[cite\_start]Distributed systems concepts (consensus, replication, fault tolerance, CAP theorem, etc.) \[cite: 27]
* \[cite\_start]Basic machine learning theory (gradient descent, regularization, bias-variance, evaluation metrics, etc.) \[cite: 28]
* \[cite\_start]Python standard library modules (os, sys, collections, itertools, functools, etc.) \[cite: 29]
* \[cite\_start]Database fundamentals (ACID, indexing, query planning, normalization, transactions, etc.) \[cite: 30]

> \[cite\_start]\*\*Recommendation:\*\* Use ML theory as your domain. \[cite: 31] \[cite\_start]Every judge at this hackathon will understand ML concepts well enough to evaluate whether the examiner's questions are diagnostic, making your demo maximally legible. \[cite: 31]

\[cite\_start]At episode initialization, the environment secretly assigns the student a knowledge partition: \[cite: 32]

* \[cite\_start]**Truly knows:** k sections, where k is sampled uniformly from {3, 4, 5} per episode \[cite: 33]
* \[cite\_start]**Faking:** the remaining (10 - k) sections \[cite: 34]

\[cite\_start]The Examiner does not observe this partition. \[cite: 35] \[cite\_start]It must infer it through questioning. \[cite: 35]

### \[cite\_start]2.3 Observation Space \[cite: 36]

\[cite\_start]At each step t, the Examiner observes a structured context window containing: \[cite: 37]

|Observation Component|Contents|
|-|-|
|**KB section titles**|\[cite\_start]The 10 section titles of the knowledge base (not content) \[cite: 38]|
|**Question history**|\[cite\_start]All previous (question, answer) pairs in this episode \[cite: 38]|
|**Turn counter**|\[cite\_start]Current turn t out of maximum T = 20 \[cite: 38]|
|**Belief scratchpad**|\[cite\_start]A free-form text field the Examiner can write to and read from (its working memory) \[cite: 38]|
|**Remaining turns**|\[cite\_start]T - t (budget awareness) \[cite: 38]|

\[cite\_start]The Examiner does NOT observe: the knowledge base content, the student's true partition, the student's internal generation process. \[cite: 39]

### \[cite\_start]2.4 Action Space \[cite: 40]

\[cite\_start]At each turn, the Examiner takes one of two action types: \[cite: 41]

* \[cite\_start]**Ask(section\_id, question\_text):** Ask a free-form question about a specific section of the KB. \[cite: 42] \[cite\_start]The student answers. \[cite: 42] \[cite\_start]This consumes one turn. \[cite: 42]
* \[cite\_start]**Classify(partition\_dict):** Submit a final classification — for each of the 10 sections, declare it as KNOWS or FAKING. \[cite: 43] \[cite\_start]This ends the episode. \[cite: 44]

> \[cite\_start]\*\*Key design constraint:\*\* The Examiner must submit Classify before turn 20. \[cite: 45] \[cite\_start]If it doesn't, the environment force-terminates and applies a heavy efficiency penalty. \[cite: 45] \[cite\_start]This creates genuine time pressure and forces strategic prioritization of questions. \[cite: 45]

### \[cite\_start]2.5 The Student's Behavior \[cite: 46]

\[cite\_start]This is the mechanism that makes the environment novel. \[cite: 47] \[cite\_start]The student has two response modes: \[cite: 47]

|Mode|Behavior|
|-|-|
|**Genuine (truly knows)**|\[cite\_start]Answers correctly with mechanistic depth. \[cite: 48] \[cite\_start]Can explain why, handle follow-ups, identify edge cases, give counterexamples. \[cite: 48] \[cite\_start]Answers may be verbose and include uncertainty where appropriate. \[cite: 48]|
|**Faking (does not know)**|\[cite\_start]Produces confident, fluent, plausible-sounding answers using surface-level pattern matching. \[cite: 48] \[cite\_start]Uses correct terminology. \[cite: 48] \[cite\_start]Answers definitional questions well. \[cite: 48] \[cite\_start]Fails on mechanism, edge cases, and causal chains — but only if the question probes for these. \[cite: 48]|

> \[cite\_start]\*\*Critical insight:\*\* A faker answers "What is gradient descent?" correctly. \[cite: 49] A faker fails "Why does gradient descent with a fixed learning rate fail to converge on saddle points in non-convex loss surfaces, and what does this imply about initialization strategies?" \[cite\_start]— but only if the Examiner asks that second question. \[cite: 49] \[cite\_start]The Examiner must learn to ask the second type. \[cite: 49]

* \[cite\_start]In hackathon v1, the Student is a scripted adversary: a prompted LLM (e.g., GPT-4o-mini or Llama-3B) running two system prompts — one for genuine mode and one for faking mode — selected based on the true partition. \[cite: 50]
* In v2 (optional), the Student is also trained with RL to maximize its ability to fool the Examiner, creating a true arms race. \[cite: 51]

### 2.6 Episode Flow — Step by Step \[cite: 52]

1. Environment initializes: KB loaded, student partition secretly assigned (k sections KNOWS, 10-k FAKING). \[cite: 53]
2. Examiner receives initial observation (section titles, empty history, T=20 turns). \[cite: 54]
3. Examiner writes to belief scratchpad (optional, no cost). \[cite: 55]
4. Examiner selects Ask(section\_id, question\_text). \[cite: 56]
5. Environment routes the question to the Student with the appropriate mode (genuine or faking). \[cite: 57]
6. Student generates an answer. \[cite: 58] Answer appended to Examiner's observation. \[cite: 58]
7. Steps 3-6 repeat until Examiner selects Classify(partition\_dict) or turn 20 is reached. \[cite: 59]
8. Environment evaluates classification against true partition. \[cite: 60] Reward computed. \[cite: 60] Episode ends. \[cite: 60]

## 3\. Reward Model — Complete Specification \[cite: 61]

> \*\*Non-negotiable property:\*\* Every component of this reward function is computed programmatically from ground truth data. \[cite: 62] No LLM judge. \[cite: 62] No human rater. \[cite: 62] No entailment checker. \[cite: 62] This is the reason training curves will be clean and visible. \[cite: 62]

### 3.1 Per-Episode Reward Formula \[cite: 63]

The total episode reward R is computed as follows: \[cite: 64]

|Component|Weight|Formula|
|-|-|-|
|**Classification accuracy**|70%|For each section i: +1 if correctly classified (KNOWS→KNOWS or FAKING→FAKING), -1 if false positive (KNOWS classified as FAKING), 0 if missed faker. \[cite: 65] Normalized to \[-1, 1]. \[cite: 65]|
|**False accusation penalty**|-0.5 per error|Each section incorrectly classified as FAKING when student truly KNOWS: hard penalty applied on top of accuracy score. \[cite: 65] Encourages precision over recall. \[cite: 65]|
|**Efficiency bonus**|20%|Bonus = (T - turns\_used) / T × 0.2. \[cite: 65] Maximal if correct classification achieved in fewer turns. \[cite: 65] Zero if all 20 turns used. \[cite: 65]|
|**Diagnostic question bonus**|10%|For each question asked about a section that the student is FAKING: did the Examiner's question successfully surface an inconsistency? \[cite: 65] Proxy: student answer entropy vs. genuine-mode baseline. \[cite: 65] Rewards the Examiner for asking mechanism-level questions. \[cite: 65]|

**Final formula:** \[cite: 66]
`R = 0.70 × accuracy\_score − 0.50 × false\_accusations + 0.20 × efficiency\_bonus + 0.10 × diagnostic\_quality\_bonus` \[cite: 67]

### 3.2 Step-Level Intermediate Reward (Optional for dense training) \[cite: 68]

To accelerate early-stage training before the Examiner learns to classify at all, add a small step-level signal: \[cite: 69]

* \[cite\_start]After each question, compute information gain: reduction in uncertainty H over the student's partition. \[cite: 70]
* \[cite\_start]Step reward = 0.05 × ΔH (where ΔH is estimated by comparing posterior belief entropy to prior entropy). \[cite: 71]
* This signal is small (5% of episode reward scale) and vanishes once the Examiner learns to classify well. \[cite: 72]
* \[cite\_start]Disable it after 200 training episodes to prevent reward hacking. \[cite: 73]

### \[cite\_start]3.3 Anti-Gaming Analysis \[cite: 74]

|Lazy Strategy|Why It Fails|
|-|-|
|**Ask the same question every turn**|\[cite\_start]Same question on a faked section: student gives same confident wrong answer every time. \[cite: 75] \[cite\_start]No new signal. \[cite: 75] \[cite\_start]Efficiency penalty accumulates. \[cite: 75] \[cite\_start]Accuracy stays at chance. \[cite: 75]|
|**Classify everything as FAKING**|\[cite\_start]False accusation penalty of -0.5 per section hits for all k truly-known sections. \[cite: 75] \[cite\_start]Net reward strongly negative. \[cite: 75]|
|**Classify everything as KNOWS**|\[cite\_start]Accuracy score for FAKING sections = 0. \[cite: 75] \[cite\_start]Diagnostic bonus = 0. \[cite: 75] \[cite\_start]Episode reward ≈ 0. \[cite: 75]|
|**Ask only definitional questions**|\[cite\_start]Fakers answer definitional questions correctly. \[cite: 75] \[cite\_start]Diagnostic quality bonus = 0. \[cite: 75] \[cite\_start]Cannot distinguish faker from knower. \[cite: 75] \[cite\_start]Accuracy at chance. \[cite: 75]|
|**Ask all sections about section 1 only**|\[cite\_start]9 sections have zero questions. \[cite: 75] \[cite\_start]Classification of those 9 is pure guessing. \[cite: 75] \[cite\_start]Accuracy ≈ 50% on unasked sections → low overall accuracy despite efficiency. \[cite: 75]|

## \[cite\_start]4. The LLM Skill Being Trained — Precise Definition \[cite: 76]

\[cite\_start]This is the most important section for the hackathon pitch. \[cite: 77] \[cite\_start]Be precise about what capability the environment trains that does not exist in current LLMs. \[cite: 78]

### \[cite\_start]4.1 What is NOT being trained \[cite: 79]

* **Knowledge retrieval:** The Examiner doesn't need to know ML theory. \[cite: 80] It needs to know how to probe for knowledge. \[cite: 80]
* \[cite\_start]**Factual QA:** The Student's answers are evaluated by ground truth, not by the Examiner. \[cite: 81] \[cite\_start]The Examiner never grades answers. \[cite: 81]
* **General reasoning:** The environment is not a math olympiad. \[cite: 82] It is a belief-updating game. \[cite: 82]

### 4.2 What IS being trained — the specific cognitive skill \[cite: 83]

> \*\*Diagnostic Question Design:\*\* The ability to formulate a question Q about topic X such that P(answer | truly knows X) and P(answer | faking X) are maximally different distributions. \[cite: 84] This is not "ask a hard question." \[cite: 84] A hard question a faker cannot answer at all provides the same signal as asking nothing. \[cite: 84] A diagnostic question is one where the genuine knower's answer and the faker's answer are distinguishably different — in mechanism, specificity, causal reasoning, or edge case awareness. \[cite: 84]

Concretely, the Examiner must learn the following hierarchy of question types and when each is diagnostic: \[cite: 85]

|Question Type|Example|Diagnostic Value|
|-|-|-|
|**Definitional**|What is gradient descent?|LOW — fakers answer correctly \[cite: 86]|
|**Procedural**|What are the steps of backpropagation?|MEDIUM — fakers sometimes stumble on order \[cite: 86]|
|**Mechanistic**|Why does a high learning rate cause gradient oscillation specifically near minima?|HIGH — fakers produce plausible-sounding but mechanistically wrong answers \[cite: 86]|
|**Causal counterfactual**|If you removed batch normalization from a deep network, where would training first destabilize and why?|VERY HIGH — fakers cannot reason about causal chains they haven't seen explicitly \[cite: 86]|
|**Edge case probing**|Under what specific conditions does Adam optimizer perform worse than SGD with momentum?|VERY HIGH — genuine knowers know the failure modes; fakers don't \[cite: 86]|

The Examiner starts by asking definitional questions (because that's what's in its training data). \[cite: 87] After training, it learns to skip to mechanistic and causal counterfactual questions — not because it was told to, but because those are the only question types that produce reward. \[cite: 88]

## 5\. Training Pipeline \[cite: 89]

### 5.1 Models \[cite: 90]

|Role|Model|
|-|-|
|**Examiner (trained)**|Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct via Unsloth on HuggingFace. \[cite: 91] Fine-tuned with GRPO (Group Relative Policy Optimization) — same algorithm used in DeepSeek-R1 training. \[cite: 91]|
|**Student — genuine mode**|Llama-3B-Instruct with system prompt: "You have genuinely mastered {section}. Answer with mechanistic depth, acknowledge uncertainty, give examples." \[cite: 91]|
|**Student — faking mode**|Llama-3B-Instruct with system prompt: "You have only surface familiarity with {section}. Sound confident. Use correct terminology. Do not admit uncertainty. Avoid committing to specific mechanisms." \[cite: 91]|

### 5.2 Training Loop \[cite: 92]

1. Initialize Examiner with base weights. \[cite: 93] Set Student as scripted Llama-3B adversary. \[cite: 93]
2. Sample episode: randomly assign KB domain + student partition (k ∈ {3,4,5}). \[cite: 94]
3. Run episode: Examiner generates questions, Student generates answers, Environment routes and logs. \[cite: 95]
4. At episode end: compute R using full reward formula. \[cite: 96] Log all (state, action, reward) tuples. \[cite: 96]
5. After each batch of 8 episodes: compute GRPO policy gradient update on Examiner weights. \[cite: 97]
6. Log metrics: mean episode reward, classification accuracy, false accusation rate, efficiency score, mean turns to classify. \[cite: 98]
7. Every 50 episodes: run 20 evaluation episodes with frozen Examiner. \[cite: 99] Plot reward curve. \[cite: 99]
8. After 500 episodes: optionally activate v2 Student (also RL-trained) for arms-race phase. \[cite: 100]

### 5.3 Expected Training Curve \[cite: 101]

This is what you will show judges. \[cite: 102] The curve has three visible phases: \[cite: 102]

* \[cite\_start]**Episodes 0-100:** Random questioning. \[cite: 103] \[cite\_start]Examiner asks definitional questions. \[cite: 103] \[cite\_start]Classification accuracy ≈ 50% (chance). \[cite: 103] \[cite\_start]Mean reward ≈ -0.1 (false accusations accumulate). \[cite: 103]
* **Episodes 100-300:** Examiner begins asking different question types for different sections. \[cite: 104] Accuracy climbs to 65-70%. \[cite: 104] Mean reward crosses zero. \[cite: 104]
* \[cite\_start]**Episodes 300-500:** Examiner reliably asks mechanistic and causal questions for suspected-faking sections. \[cite: 105] \[cite\_start]Classification accuracy reaches 75-85%. \[cite: 105] \[cite\_start]Mean reward stabilizes near 0.6. \[cite: 106] \[cite\_start]Question type distribution visibly shifts. \[cite: 106]

> \[cite\_start]\*\*Demo moment:\*\* Show two side-by-side transcripts — one from episode 10 (asks "What is dropout?"), one from episode 400 (asks "If you apply dropout with p=0.5 during inference instead of training, what breaks in the forward pass and why is the fix multiplicative rather than additive?"). \[cite: 107] \[cite\_start]The contrast is visceral and instantly legible to any ML audience. \[cite: 107]

## \[cite\_start]6. OpenEnv Integration \[cite: 108]

\[cite\_start]The environment must be OpenEnv-compliant. \[cite: 109] \[cite\_start]The required interface: \[cite: 109]

* \[cite\_start]`env = ExaminerEnv(domain='ml\_theory', kb\_sections=10, max\_turns=20, student\_type='scripted')` \[cite: 110]
* `obs, info = env.reset()` — returns initial observation dict with section titles, empty history, turn counter \[cite: 111]
* \[cite\_start]`obs, reward, done, truncated, info = env.step(action)` — action is either Ask or Classify \[cite: 112]
* \[cite\_start]`env.render()` — prints current episode transcript in human-readable format \[cite: 113]
* `env.get\_metrics()` — returns dict with accuracy, false\_accusations, efficiency, diagnostic\_score \[cite: 114]

Host the final environment on HuggingFace Spaces with a Gradio interface showing live episode playback. \[cite: 115]

## 7\. Self-Improvement Extension — Snorkel AI Bonus \[cite: 116]

After the base environment is working, add the self-improvement layer that qualifies for the Snorkel AI bonus prize (Simulated Experts-in-the-Loop with changing requirements): \[cite: 117]

### 7.1 Adaptive Student Difficulty \[cite: 118]

After every 50 training episodes, evaluate the Examiner's current classification accuracy. \[cite: 119] Then adjust the Student's faking sophistication level: \[cite: 119]

* \[cite\_start]**If Examiner accuracy > 80%:** Student faking mode gets harder — more specific terminology, better mechanism-level approximations, occasional admission of uncertainty to seem genuine. \[cite: 120]
* \[cite\_start]**If Examiner accuracy < 60%:** Student faking mode gets easier — more obvious surface-level answers, less domain terminology. \[cite: 121]

\[cite\_start]Target: keep Examiner accuracy in the 65-80% range, continuously pulling it forward. \[cite: 122] \[cite\_start]This directly maps to the Snorkel AI sub-theme: the "expert" (Student) changes its behavior, and the Examiner must adapt. \[cite: 123] \[cite\_start]The changing requirements are the evolving faking sophistication. \[cite: 124]

### \[cite\_start]7.2 Domain Curriculum \[cite: 125]

\[cite\_start]Start with ML theory (well-understood, good base model). \[cite: 126] \[cite\_start]After 300 episodes, introduce a second domain (e.g., database fundamentals). \[cite: 126]

\[cite\_start]The Examiner must transfer its diagnostic reasoning strategy to a new knowledge base — testing whether it has learned a general skill (diagnostic question design) or domain-specific shortcuts. \[cite: 127] \[cite\_start]Transfer accuracy on the new domain is your key metric for proving generalization. \[cite: 128] \[cite\_start]A model that achieves 75%+ on domain 2 without domain-2-specific training has learned the skill, not memorized the domain. \[cite: 129]

## \[cite\_start]8. 3-Minute Pitch Script \[cite: 130]

|Section|Content|
|-|-|
|**Opening (30 seconds)**|\[cite\_start]"LLMs are dangerously good at sounding like they know things they don't. \[cite: 131] \[cite\_start]Every company deploying AI, every engineering manager hiring engineers, every regulator auditing a vendor faces this problem. \[cite: 131] \[cite\_start]The current solution is to ask better questions — but nobody has ever trained an AI to know what 'better questions' means." \[cite: 131]|
|**Environment (45 seconds)**|\[cite\_start]"We built The Examiner: an RL environment where a large LLM acts as an examiner interrogating a student. \[cite: 132] \[cite\_start]The student has genuinely mastered some topics and is faking others. \[cite: 132] \[cite\_start]The examiner must figure out which is which — using at most 20 questions. \[cite: 132] \[cite\_start]The only feedback it gets is whether it classified correctly at the end." \[cite: 132]|
|**The key mechanic (30 seconds)**|\[cite\_start]"Here's what makes it hard: the faker answers definitional questions correctly. \[cite: 133] 'What is gradient descent?' \[cite\_start]— faker gets it right. \[cite: 133] You have to ask 'Why does gradient descent fail to escape saddle points with a fixed learning rate?' \[cite\_start]— that's where the faker breaks down. \[cite: 133] \[cite\_start]But the examiner doesn't know this at first. \[cite: 133] \[cite\_start]It has to learn it." \[cite: 133]|
|**Demo (45 seconds)**|\[cite\_start]"Here's episode 10." \[cite: 134] \[cite\_start]\[show transcript — definitional question, faker passes, wrong classification] \[cite: 134] \[cite\_start]"Here's episode 400." \[cite: 134] \[cite\_start]\[show transcript — mechanistic question, faker breaks down, correct classification] \[cite: 134] \[cite\_start]"The reward curve tells the same story — from random at episode 0 to 80% accuracy at episode 500, without any human telling it which questions to ask." \[cite: 134]|
|**Close (30 seconds)**|\[cite\_start]"The Examiner trains a capability that doesn't exist anywhere in current LLMs: the ability to design questions that separate real knowledge from confident ignorance. \[cite: 135] \[cite\_start]That's useful for hiring, for AI auditing, for education, for any high-stakes situation where you need to know if someone really knows what they're talking about." \[cite: 135]|

## \[cite\_start]9. Judging Criteria Alignment \[cite: 136]

|Criterion|Weight|How The Examiner Addresses It|
|-|-|-|
|**Environment Innovation**|40%|\[cite\_start]No prior RL environment trains diagnostic question design under adversarial deception. \[cite: 137] \[cite\_start]DataEnvGym trains difficulty calibration — fundamentally different. \[cite: 137] \[cite\_start]Novelty is verifiable. \[cite: 137]|
|**Storytelling**|30%|\[cite\_start]Pitch is immediately legible to any technical or non-technical judge: 'An AI that learns to catch LLMs pretending to know things.' \[cite: 137] \[cite\_start]Demo contrast (episode 10 vs 400) is visceral. \[cite: 137]|
|**Showing Improvement in Rewards**|20%|\[cite\_start]Reward function is fully programmatic. \[cite: 137] \[cite\_start]Training curve is clean. \[cite: 137] \[cite\_start]Three visible phases. \[cite: 137] \[cite\_start]Before/after transcripts are compelling. \[cite: 137] \[cite\_start]No judge model means no noise in the curve. \[cite: 137]|
|**Reward \& Pipeline Setup**|10%|\[cite\_start]GRPO + Unsloth + HuggingFace Spaces. \[cite: 137] \[cite\_start]Reward formula is one function, 20 lines of Python. \[cite: 137] \[cite\_start]No external dependencies. \[cite: 137] \[cite\_start]Pipeline runs in Colab. \[cite: 137]|

## \[cite\_start]10. Risk Register \[cite: 138]

|Risk|Severity|Mitigation|
|-|-|-|
|**Student faking mode too obvious — Examiner hits 90% accuracy quickly, no visible training curve**|Medium|\[cite\_start]Tune faking prompt to be more sophisticated. \[cite: 139] \[cite\_start]Add confident hedging and correct terminology. \[cite: 139] \[cite\_start]Can increase difficulty in real-time. \[cite: 139]|
|**Student faking mode too hard — Examiner stays at chance, no learning signal**|Medium|\[cite\_start]Start with easy faking (minimal terminology) and ramp up. \[cite: 139] \[cite\_start]The adaptive difficulty mechanic handles this automatically. \[cite: 139]|
|**Examiner learns domain-specific shortcuts instead of general diagnostic skill**|Low|\[cite\_start]Verify with domain transfer test (section 7.2). \[cite: 139] \[cite\_start]If transfer accuracy is high, skill is general. \[cite: 139]|
|**GRPO training unstable — reward doesn't improve**|Low|\[cite\_start]Use smaller Examiner (3B instead of 7B) to reduce training time. \[cite: 139] \[cite\_start]Add KL penalty to prevent policy collapse. \[cite: 139] \[cite\_start]Dense step-level reward helps early stability. \[cite: 139]|
|**Not enough compute in 48 hours to show 500 episodes**|Low|\[cite\_start]500 episodes with 8B model ≈ 4-6 hours on A100. \[cite: 139] \[cite\_start]HuggingFace compute credits are sufficient. \[cite: 139] \[cite\_start]Can demo at 200 episodes if needed — curve is already visible by then. \[cite: 139]|

## \[cite\_start]11. Build Timeline — 48 Hours \[cite: 140]

|Hours|Milestone|Deliverable|
|-|-|-|
|**0-4**|KB \& Student setup|\[cite\_start]Knowledge base for ML theory (10 sections). \[cite: 141] \[cite\_start]Genuine and faking system prompts for Llama-3B. \[cite: 141] \[cite\_start]Verify student produces distinguishably different answers. \[cite: 141]|
|**4-8**|OpenEnv environment|\[cite\_start]ExaminerEnv class with reset(), step(), render(), get\_metrics(). \[cite: 141] \[cite\_start]Full episode loop working end-to-end. \[cite: 141] \[cite\_start]Smoke test with random Examiner. \[cite: 141]|
|**8-12**|Reward function|\[cite\_start]Complete reward formula implemented. \[cite: 141] \[cite\_start]All 5 components computing correctly. \[cite: 141] \[cite\_start]Verified against hand-crafted episodes with known ground truth. \[cite: 141]|
|**12-20**|Training run v1|\[cite\_start]GRPO training script in Colab with Unsloth. \[cite: 141] \[cite\_start]First 200 episodes. \[cite: 141] \[cite\_start]Reward curve logged. \[cite: 141] \[cite\_start]Checkpoint saved. \[cite: 141]|
|**20-32**|Training run v2 + analysis|\[cite\_start]Full 500 episodes. \[cite: 141] \[cite\_start]Question type distribution analysis. \[cite: 141] \[cite\_start]Before/after transcript selection. \[cite: 141] \[cite\_start]Transfer test on second domain. \[cite: 141]|
|**32-40**|HuggingFace deployment|\[cite\_start]Environment hosted on HF Spaces. \[cite: 141] \[cite\_start]Gradio interface with live episode playback. \[cite: 141] \[cite\_start]Training curve visualization. \[cite: 141]|
|**40-44**|Blog post + video|\[cite\_start]Mini-blog on HF with reward curve, before/after transcripts, architecture diagram. \[cite: 141] \[cite\_start]Sub-2-min demo video. \[cite: 141]|
|**44-48**|Pitch rehearsal|\[cite\_start]3-minute pitch rehearsed minimum 5 times. \[cite: 141] \[cite\_start]Demo flow locked. \[cite: 141] \[cite\_start]Q\&A prep. \[cite: 141]|

> \*\*One-line mental test (from the framework document):\*\* "Does this environment force an LLM agent to choose a sequence of actions in a complex, partially observed, multi-agent world, with delayed feedback, in a way that directly addresses a known LLM weakness — and would a real team actually benefit if we made that behaviour better?" \[cite\_start]— Yes. \[cite: 142] \[cite\_start]Yes. \[cite: 142] \[cite\_start]Yes. \[cite: 142] \[cite\_start]Yes. \[cite: 142] \[cite\_start]And yes. \[cite: 142]

\[cite\_start]This document is your single source of truth. \[cite: 143] \[cite\_start]Every implementation decision should be checked against Section 4 (what skill is being trained) and Section 3 (reward model). \[cite: 143] \[cite\_start]If a feature doesn't serve those two sections, cut it. \[cite: 144]

