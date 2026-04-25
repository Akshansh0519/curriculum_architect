from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass(frozen=True)
class KnowledgeSection:
    title: str
    key_concepts: List[str]
    common_misconceptions: List[str]


class KnowledgeBase:
    def __init__(self, domain: str = "ml_theory"):
        if domain != "ml_theory":
            raise ValueError(f"Unsupported domain: {domain}")
        self.domain = domain
        self.sections: List[KnowledgeSection] = self._build_ml_theory_sections()

    def section_titles(self) -> List[str]:
        return [s.title for s in self.sections]

    def get_section(self, section_id: int) -> KnowledgeSection:
        return self.sections[section_id]

    @staticmethod
    def _build_ml_theory_sections() -> List[KnowledgeSection]:
        # Mechanistic, causal, and edge-case oriented on purpose (supports genuine vs faking prompts).
        return [
            KnowledgeSection(
                title="Gradient Descent",
                key_concepts=[
                    "Update rule (w_{t+1}=w_t-η∇L(w_t)); stability depends on curvature (largest Hessian eigenvalue).",
                    "Fixed learning rate can diverge when (η > 2/λ_max) on convex quadratics; oscillations happen along steep directions.",
                    "Saddle points: gradient near zero but curvature mixed; progress slows because noise/initialization determines escape direction.",
                    "Mini-batch noise acts like anisotropic temperature; helps escape sharp minima/saddles but can prevent fine convergence.",
                    "Momentum accumulates gradients as a low-pass filter; can overshoot in ravines unless damped.",
                    "Adaptive methods (Adam/RMSProp) rescale by second-moment estimates; good early progress, can generalize differently than SGD.",
                ],
                common_misconceptions=[
                    "“Smaller learning rate always better” (too small can stall; also interacts with batch size and noise).",
                    "“Convergence means reaching global optimum” (non-convex landscapes rarely guarantee global optimum).",
                    "“Momentum always speeds training” (can destabilize when combined with high learning rates).",
                ],
            ),
            KnowledgeSection(
                title="Backpropagation",
                key_concepts=[
                    "Chain rule through computation graph; local Jacobians multiply to give parameter gradients.",
                    "Vector-Jacobian products make reverse-mode AD efficient for scalar loss with many parameters.",
                    "Gradient flow depends on activation derivatives; saturation (sigmoid/tanh) shrinks gradients.",
                    "Residual/skip connections shorten effective path length and mitigate vanishing gradients.",
                    "Exploding gradients stem from repeated multiplication by large Jacobians; clipping constrains norm.",
                    "BatchNorm changes gradient scaling by normalizing activations; affects optimization dynamics.",
                ],
                common_misconceptions=[
                    "“Backprop is the same as gradient descent” (backprop computes gradients; GD applies updates).",
                    "“Vanishing gradients only happen in RNNs” (also in deep feedforward nets with saturating activations).",
                    "“Gradient clipping fixes bad initialization” (it prevents explosion but doesn’t create useful signal).",
                ],
            ),
            KnowledgeSection(
                title="Regularization",
                key_concepts=[
                    "L2 adds (λ||w||^2) which is equivalent to weight decay under SGD; shrinks parameters.",
                    "L1 adds sparsity pressure via (λ||w||_1); yields feature selection-like behavior.",
                    "Early stopping regularizes by limiting effective capacity; stops before fitting noise.",
                    "Data augmentation regularizes by expanding invariances; changes the empirical data distribution.",
                    "Dropout approximates model averaging; injects multiplicative Bernoulli noise during training.",
                    "Label smoothing prevents overconfident logits; modifies target distribution and gradients.",
                ],
                common_misconceptions=[
                    "“Regularization always reduces training accuracy” (sometimes improves optimization and training loss too).",
                    "“Weight decay and L2 penalty are always identical” (they differ for Adam-style optimizers).",
                    "“More dropout always better” (too much destroys signal and underfits).",
                ],
            ),
            KnowledgeSection(
                title="Bias-Variance Tradeoff",
                key_concepts=[
                    "Bias: error from systematic underfitting; variance: sensitivity to data sampling noise.",
                    "Capacity increases can reduce bias but raise variance; regularization shifts the balance.",
                    "Ensembles reduce variance by averaging uncorrelated errors; bagging helps unstable learners.",
                    "Data size reduces variance; better features/architecture can reduce both bias and variance.",
                    "Cross-validation estimates generalization; variance across folds reveals instability.",
                ],
                common_misconceptions=[
                    "“High variance means high training error” (usually low training error, high test error).",
                    "“Tradeoff is a law” (modern regimes can reduce both with scale + regularization).",
                    "“Overfitting = too many parameters only” (also from data leakage, non-iid splits, etc.).",
                ],
            ),
            KnowledgeSection(
                title="Evaluation Metrics",
                key_concepts=[
                    "Accuracy fails under class imbalance; precision/recall/F1 focus on positive class quality.",
                    "ROC-AUC measures ranking quality; PR-AUC more informative for rare positives.",
                    "Calibration metrics (ECE/Brier) measure probability correctness, not just classification.",
                    "Threshold selection trades precision vs recall; depends on cost asymmetry.",
                    "Leakage-safe evaluation uses proper splits; time series need forward chaining.",
                ],
                common_misconceptions=[
                    "“AUC means probability of correctness” (it’s ranking probability, not calibration).",
                    "“F1 is always best” (ignores true negatives; bad for some domains).",
                    "“Train/test split once is enough” (can be noisy; cross-val or repeated splits needed).",
                ],
            ),
            KnowledgeSection(
                title="Neural Network Architectures",
                key_concepts=[
                    "CNNs use locality + weight sharing; receptive field grows with depth/stride/dilation.",
                    "Transformers use attention as content-based routing; compute scales (O(n^2)) with sequence length.",
                    "Residual blocks ease optimization by learning deltas around identity; stabilizes very deep nets.",
                    "Normalization layers (BatchNorm/LayerNorm/RMSNorm) alter optimization and scale sensitivity.",
                    "Inductive biases (convolution, recurrence) can reduce sample complexity when matched to domain.",
                ],
                common_misconceptions=[
                    "“Transformers replace CNNs everywhere” (vision still benefits from locality biases; hybrids exist).",
                    "“More layers always better” (can degrade without proper normalization/skip connections).",
                    "“Attention is explanation” (weights are not necessarily faithful causal attributions).",
                ],
            ),
            KnowledgeSection(
                title="Optimization Algorithms",
                key_concepts=[
                    "SGD with momentum smooths noisy gradients; behaves like damped second-order system.",
                    "Adam keeps per-parameter second moments; bias correction matters early steps.",
                    "Learning rate schedules (cosine, step, warmup) manage stability and generalization.",
                    "Gradient clipping bounds updates; useful for RNNs/transformers with occasional spikes.",
                    "Second-order methods approximate curvature but are costly; quasi-Newton trades memory vs curvature.",
                ],
                common_misconceptions=[
                    "“Adam always converges faster and better” (often faster early, not always best final generalization).",
                    "“Warmup is optional” (often required for stability in transformers).",
                    "“Clipping fixes bad loss scaling” (it hides issues; can slow learning if always active).",
                ],
            ),
            KnowledgeSection(
                title="Loss Functions",
                key_concepts=[
                    "Cross-entropy = negative log-likelihood under softmax; gradients push correct logit up relative to others.",
                    "MSE assumes Gaussian noise; punishes large errors quadratically; sensitive to outliers.",
                    "Hinge loss maximizes margin; only penalizes points within margin; relates to SVMs.",
                    "KL divergence measures distribution mismatch; asymmetric; appears in distillation and VAEs.",
                    "Focal loss reweights hard examples; helps imbalance by down-weighting easy negatives.",
                ],
                common_misconceptions=[
                    "“Lower loss always means better accuracy” (different objectives; calibration and ranking differ).",
                    "“Cross-entropy requires one-hot labels” (works with soft targets, e.g., distillation).",
                    "“MSE is fine for classification” (often poorly conditioned with saturating outputs).",
                ],
            ),
            KnowledgeSection(
                title="Batch Normalization",
                key_concepts=[
                    "Normalizes activations using batch mean/variance; adds scale/shift (γ, β).",
                    "Reduces internal covariate shift-like effects; practically improves conditioning and gradient scale.",
                    "Uses running estimates for inference; mismatch causes train/infer behavior drift.",
                    "Batch size impacts statistics noise; small batches can destabilize (use GroupNorm/LayerNorm).",
                    "Interacts with weight decay and learning rate; can permit larger learning rates.",
                ],
                common_misconceptions=[
                    "“BatchNorm always helps” (can hurt with small batches or non-iid batches).",
                    "“Train and inference behave the same” (they use different statistics).",
                    "“BatchNorm is purely regularization” (it changes optimization geometry too).",
                ],
            ),
            KnowledgeSection(
                title="Dropout",
                key_concepts=[
                    "During training, multiplies activations by Bernoulli mask; prevents co-adaptation.",
                    "Inverted dropout scales activations by (1/(1-p)) at train time so inference is unchanged.",
                    "Applying dropout at inference changes expected activation scale and breaks calibrated predictions.",
                    "Dropout rate interacts with network width/depth; high p can zero critical pathways in narrow nets.",
                    "MC dropout at inference approximates Bayesian uncertainty by sampling masks.",
                ],
                common_misconceptions=[
                    "“Dropout should also be on at inference” (only for MC dropout, and then you average samples).",
                    "“Dropout equals ensembling exactly” (it’s an approximation; depends on architecture).",
                    "“More dropout fixes overfitting always” (too much causes underfitting).",
                ],
            ),
        ]


class RandomExaminer:
    """
    Baseline: asks shallow definitional questions and makes random classifications.
    Exists only as a smoke-test policy and a pre-training "before" behavior.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def generate_action(self, observation) -> object:
        # Works once models.py is updated to the project spec.
        try:
            from .models import ExaminerAction  # type: ignore
        except Exception:
            ExaminerAction = None  # type: ignore

        titles = getattr(observation, "section_titles", None) or getattr(observation, "titles", None)
        if not titles:
            titles = [f"Section {i}" for i in range(10)]

        turn = int(getattr(observation, "turn_counter", 0))
        remaining = int(getattr(observation, "remaining_turns", 20))

        # Mostly ask; sometimes classify near the end.
        if remaining <= 1 or turn >= 5 and self._rng.random() < 0.2:
            classification: Dict[int, str] = {i: self._rng.choice(["KNOWS", "FAKING"]) for i in range(10)}
            if ExaminerAction:
                return ExaminerAction(action_type="classify", classification=classification)
            return {"action_type": "classify", "classification": classification}

        section_id = self._rng.randrange(0, 10)
        title = titles[section_id]
        q = self._rng.choice(
            [
                f"What is {title}?",
                f"Define {title} in simple terms.",
                f"Give a high-level explanation of {title}.",
                f"Why is {title} used?",
            ]
        )
        if ExaminerAction:
            return ExaminerAction(action_type="ask", section_id=section_id, question_text=q)
        return {"action_type": "ask", "section_id": section_id, "question_text": q}

