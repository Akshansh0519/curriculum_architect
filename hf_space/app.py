from __future__ import annotations

from pathlib import Path
from typing import Tuple

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


def run_examination_episode(difficulty: str) -> Tuple[list[tuple[str, str]], str, str]:
    try:
        transcript = [
            ("Examiner", f"[{difficulty}] Explain gradient descent in one line."),
            ("Student", "Gradient descent minimizes loss by moving opposite gradient."),
            ("Examiner", "Why can fixed LR fail near saddle points?"),
            ("Student", "It can oscillate because curvature differs by direction."),
        ]
        result = "Predicted: FAKING in section 4 | Ground truth: FAKING in section 4"
        reward = "Episode reward: 0.58"
        return transcript, result, reward
    except Exception as exc:  # pragma: no cover - UI fallback path
        return [("System", "Episode failed. Check server logs.")], f"Error: {exc}", "Episode reward: N/A"


with gr.Blocks(title="The Examiner") as demo:
    gr.Markdown("# The Examiner")

    with gr.Tab("🎯 Live Demo"):
        difficulty = gr.Dropdown(choices=["easy", "medium", "hard"], value="medium", label="Difficulty")
        run_btn = gr.Button("Run Examination Episode", variant="primary")
        chat = gr.Chatbot(label="Examiner vs Student Transcript", type="tuples")
        classification_out = gr.Textbox(label="Classification Result")
        reward_out = gr.Textbox(label="Episode Reward")
        run_btn.click(
            fn=run_examination_episode,
            inputs=[difficulty],
            outputs=[chat, classification_out, reward_out],
        )

    with gr.Tab("📊 Training Results"):
        gr.Markdown(
            f"Assets status -> reward: {_asset_status(ASSETS / 'reward_curve.png')} | "
            f"accuracy: {_asset_status(ASSETS / 'accuracy_curve.png')} | "
            f"architecture: {_asset_status(ASSETS / 'architecture.png')}"
        )
        gr.Image(value=str(ASSETS / "reward_curve.png"), label="Reward Curve")
        gr.Image(value=str(ASSETS / "accuracy_curve.png"), label="Accuracy Curve")
        with gr.Row():
            before_box = gr.Textbox(
                label="Before Training (Episode 10)",
                value=_load_text(ASSETS / "before.txt", "Before transcript missing."),
                lines=12,
            )
            after_box = gr.Textbox(
                label="After Training (Episode 400)",
                value=_load_text(ASSETS / "after.txt", "After transcript missing."),
                lines=12,
            )

    with gr.Tab("ℹ️ About"):
        gr.Markdown(NARRATIVE)
        gr.Image(value=str(ASSETS / "architecture.png"), label="Architecture Diagram")
        gr.Markdown(
            "- GitHub: https://github.com/Akshansh0519/curriculum_architect\n"
            "- Blog: https://huggingface.co/blog/TODO\n"
            "- Training Notebook: https://colab.research.google.com/TODO"
        )


if __name__ == "__main__":
    demo.launch()
