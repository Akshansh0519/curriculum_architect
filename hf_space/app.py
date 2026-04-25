from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Tuple

import gradio as gr

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

NARRATIVE = (
    "LLMs are dangerously good at sounding like they know things they don't — and current AI systems "
    "cannot tell the difference. The Examiner is an RL environment that trains an AI to design diagnostic "
    "questions that reliably separate genuine knowledge from confident faking. After training, the Examiner "
    "learns to skip surface-level questions and ask the mechanistic, causal questions that only genuine experts "
    "can answer."
)


def _load_text(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def _asset_status(path: Path) -> str:
    return "Ready" if path.exists() and path.stat().st_size > 0 else f"Missing: {path.name}"


def _run_real_episode(difficulty: str) -> Tuple[List[Tuple[str, str]], str, str]:
    """Run a real ExaminerEnvironment episode."""
    try:
        from examiner_env.client import run_local_episode

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(run_local_episode(difficulty=difficulty))
        loop.close()

        chat_history: List[Tuple[str, str]] = []
        transcript = result.get("transcript", [])
        for turn_a, turn_b in zip(transcript[::2], transcript[1::2]):
            chat_history.append((turn_a.get("text", ""), turn_b.get("text", "")))

        accuracy = result.get("accuracy", 0.0)
        false_acc = int(result.get("false_accusations", 0))
        turns = int(result.get("turns_used", 0))
        reward = result.get("reward", 0.0)

        classification_text = (
            f"Classification submitted | Accuracy: {accuracy:.0%} | "
            f"False accusations: {false_acc} | Turns used: {turns}"
        )
        reward_text = f"Episode reward: {reward:.4f}"
        return chat_history, classification_text, reward_text

    except Exception as exc:
        return (
            [("System", f"Live env unavailable: {exc}. Showing demo transcript.")],
            _demo_classification(difficulty),
            "Episode reward: demo mode",
        )


def _demo_classification(difficulty: str) -> str:
    return f"[{difficulty}] Predicted: FAKING in section 4 | Ground truth: FAKING in section 4"


def run_examination_episode(difficulty: str) -> Tuple[List[Tuple[str, str]], str, str]:
    return _run_real_episode(difficulty)


with gr.Blocks(title="The Examiner") as demo:
    gr.Markdown("# 🧠 The Examiner")
    gr.Markdown(
        "_An RL environment that trains an AI to design diagnostic questions "
        "that reliably separate genuine knowledge from confident faking._"
    )

    with gr.Tab("🎯 Live Demo"):
        gr.Markdown("Run a live examination episode. The Examiner asks questions; the Student answers.")
        difficulty = gr.Dropdown(
            choices=["easy", "medium", "hard"], value="medium", label="Difficulty"
        )
        run_btn = gr.Button("▶ Run Examination Episode", variant="primary")
        chat = gr.Chatbot(
            label="Examiner ↔ Student Transcript",
            type="tuples",
            height=400,
        )
        with gr.Row():
            classification_out = gr.Textbox(label="Classification Result", lines=2)
            reward_out = gr.Textbox(label="Episode Reward", lines=2)
        run_btn.click(
            fn=run_examination_episode,
            inputs=[difficulty],
            outputs=[chat, classification_out, reward_out],
        )

    with gr.Tab("📊 Training Results"):
        gr.Markdown(
            f"**Asset status** → reward_curve: {_asset_status(ASSETS / 'reward_curve.png')} | "
            f"accuracy_curve: {_asset_status(ASSETS / 'accuracy_curve.png')} | "
            f"architecture: {_asset_status(ASSETS / 'architecture.png')}"
        )
        with gr.Row():
            gr.Image(value=str(ASSETS / "reward_curve.png"), label="Reward Curve (Episodes vs Mean Reward)")
            gr.Image(value=str(ASSETS / "accuracy_curve.png"), label="Accuracy Curve")
        gr.Markdown("### Before vs After Training")
        with gr.Row():
            gr.Textbox(
                label="Before Training (Episode ~10)",
                value=_load_text(ASSETS / "before.txt", "Before transcript not yet available."),
                lines=10,
            )
            gr.Textbox(
                label="After Training (Episode ~400)",
                value=_load_text(ASSETS / "after.txt", "After transcript not yet available."),
                lines=10,
            )

    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"## Project Narrative\n\n{NARRATIVE}")
        gr.Image(value=str(ASSETS / "architecture.png"), label="System Architecture")
        gr.Markdown(
            "### Links\n"
            "- **GitHub:** https://github.com/Akshansh0519/curriculum_architect\n"
            "- **Blog / Writeup:** https://huggingface.co/blog/TODO\n"
            "- **Training Notebook:** https://colab.research.google.com/TODO\n"
            "- **Demo Video:** https://youtube.com/TODO"
        )
        gr.Markdown(
            "### Tech Stack\n"
            "| Component | Technology |\n"
            "|---|---|\n"
            "| RL Algorithm | GRPO via TRL GRPOTrainer |\n"
            "| Examiner Model | Qwen2.5-7B-Instruct (Unsloth LoRA) |\n"
            "| Student Model | Llama-3.2-3B-Instruct |\n"
            "| Environment | OpenEnv (openenv-core) |\n"
            "| Logging | Weights & Biases |\n"
            "| Deployment | Gradio on HuggingFace Spaces |"
        )


if __name__ == "__main__":
    demo.launch()
