# CONTEXT PRIMER — The Examiner
# Paste this into your AI tool at every session start. Target: <200 tokens.

Project: The Examiner — RL env training diagnostic question design under adversarial deception.
Stack: openenv-core (latest), Unsloth (≥2025.4), TRL GRPOTrainer (≥0.15), Qwen2.5-7B-Instruct, Llama-3.2-3B-Instruct, wandb (≥0.19), gradio (≥5.0), HF Hub (≥0.27), Python 3.10+.
C1 owns: examiner_env/ (models, reward, KB, student, server). C2 owns: training/, scripts/, hf_space/, client.py. VAL owns: tests/.
Current phase: [UPDATE]
Open MSRs: [UPDATE — e.g., MSR-1,2,3,4,5,6,7,8,9]
Key constraints: async methods only, GRPOTrainer only (no custom loops), programmatic reward only (no LLM judge), no local paths in Colab, no video files in HF repo.
Reward: R = 0.70*acc - 0.50*FA + 0.20*eff + 0.10*diag.
Full rules: guardrails.md.
