"""A small Gradio demo comparing FGSM and PGD attacks on ImageNet images."""

from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = ResNet18_Weights.DEFAULT
LABELS = WEIGHTS.meta["categories"]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


@lru_cache(maxsize=1)
def get_model() -> torch.nn.Module:
    """Download once, cache, and return an evaluation-mode ImageNet model."""
    return resnet18(weights=WEIGHTS).to(DEVICE).eval()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Apply the spatial part of ImageNet preprocessing, preserving pixel scale."""
    image = image.convert("RGB")
    image = TF.resize(image, 256, interpolation=InterpolationMode.BILINEAR)
    image = TF.center_crop(image, [224, 224])
    return TF.to_tensor(image).unsqueeze(0).to(DEVICE)


def logits_for(model: torch.nn.Module, pixels: torch.Tensor) -> torch.Tensor:
    return model((pixels - MEAN) / STD)


def prediction(logits: torch.Tensor) -> tuple[int, float]:
    probabilities = logits.softmax(dim=1)
    confidence, class_id = probabilities.max(dim=1)
    return class_id.item(), confidence.item()


def fgsm_attack(
    model: torch.nn.Module, pixels: torch.Tensor, label: int, epsilon: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Maximize loss for the original prediction with one signed-gradient step."""
    attacked = pixels.detach().clone().requires_grad_(True)
    logits = logits_for(model, attacked)
    target = torch.tensor([label], device=DEVICE)
    loss = torch.nn.functional.cross_entropy(logits, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient = attacked.grad.detach()
    adversarial = (attacked + epsilon * gradient.sign()).clamp(0, 1).detach()
    return adversarial, gradient


def pgd_attack(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    label: int,
    epsilon: float,
    step_size: float,
    steps: int,
) -> torch.Tensor:
    """Iteratively maximize loss while staying inside an L-infinity pixel budget."""
    original = pixels.detach()
    adversarial = original.clone()
    target = torch.tensor([label], device=DEVICE)

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(
            logits_for(model, adversarial), target
        )
        gradient = torch.autograd.grad(loss, adversarial)[0]

        # Take a small FGSM-like step, then project back into the allowed region.
        adversarial = adversarial.detach() + step_size * gradient.sign()
        lower = original - epsilon
        upper = original + epsilon
        adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
        adversarial = adversarial.clamp(0, 1)

    return adversarial.detach()


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return np.uint8(np.clip(array * 255.0, 0, 255))


def run_demo(
    image: Image.Image | None,
    epsilon: float,
    pgd_steps: int,
    pgd_step_size: float,
):
    if image is None:
        raise gr.Error("Please upload an image first.")

    model = get_model()
    pixels = image_to_tensor(image)

    with torch.no_grad():
        clean_logits = logits_for(model, pixels)
    clean_id, clean_confidence = prediction(clean_logits)

    epsilon = float(epsilon)
    fgsm_image, _ = fgsm_attack(model, pixels, clean_id, epsilon)
    with torch.no_grad():
        fgsm_logits = logits_for(model, fgsm_image)
    fgsm_id, fgsm_confidence = prediction(fgsm_logits)

    pgd_image = pgd_attack(
        model,
        pixels,
        clean_id,
        epsilon,
        float(pgd_step_size),
        int(pgd_steps),
    )
    with torch.no_grad():
        pgd_logits = logits_for(model, pgd_image)
    pgd_id, pgd_confidence = prediction(pgd_logits)

    # Amplify the tiny pixel differences so students can inspect their structure.
    delta = pgd_image - pixels
    amplified = (delta / (2 * max(epsilon, 1e-6)) + 0.5).clamp(0, 1)
    report = (
        f"### Results\n"
        f"- **Original:** {LABELS[clean_id]} ({clean_confidence:.1%})\n"
        f"- **After FGSM:** {LABELS[fgsm_id]} ({fgsm_confidence:.1%}) "
        f"— changed: {'yes' if clean_id != fgsm_id else 'no'}\n"
        f"- **After PGD:** {LABELS[pgd_id]} ({pgd_confidence:.1%}) "
        f"— changed: {'yes' if clean_id != pgd_id else 'no'}\n"
        f"- **Budget:** ε = {epsilon:.3f} (about {epsilon * 255:.1f}/255); "
        f"PGD used {int(pgd_steps)} steps of {float(pgd_step_size):.3f}"
    )
    return (
        report,
        to_numpy_image(fgsm_image),
        to_numpy_image(pgd_image),
        to_numpy_image(amplified),
    )


TUTORIAL = r"""
<div class="tutorial-kicker">INTERACTIVE COMPANION</div>

# From pixels to adversarial image

Follow the four stages in the same order as the program. You can leave this
guide open while adjusting the attack controls behind it.

## 01 · Prepare the image

```python
pixels = image_to_tensor(image)
```

Resizes and center-crops the image to 224 × 224, converts it to RGB numbers in
the range 0–1, and adds a one-image batch dimension.

```python
clean_logits = model((pixels - MEAN) / STD)
```

Normalizes each color channel the way ResNet-18 expects, then asks the model for
1,000 ImageNet class scores (logits).

```python
clean_id, clean_confidence = prediction(clean_logits)
```

Turns the scores into probabilities and records the model's original answer.
That class becomes the label both attacks try to move away from.

## 02 · FGSM — one large step

```python
attacked.requires_grad_(True)
```

Tells PyTorch to track how the loss changes when each input pixel changes.

```python
loss = cross_entropy(logits, target)
```

Measures how well the model still supports its original prediction.

```python
loss.backward()
```

Computes one gradient value for every input pixel and color channel.

```python
direction = gradient.sign()
```

Keeps only whether each gradient points up or down. This spends the pixel budget
in the most loss-increasing direction under an L∞ constraint.

```python
adversarial = attacked + epsilon * direction
```

Makes FGSM's single step. Epsilon is the maximum permitted change to any channel.

```python
adversarial = adversarial.clamp(0, 1)
```

Keeps the result inside the valid pixel range.

## 03 · PGD — several small steps

```python
for _ in range(steps):
```

Repeats the gradient calculation. Unlike FGSM, PGD can adjust its direction as
the image moves through the model's decision landscape.

```python
adversarial += step_size * gradient.sign()
```

Takes one small loss-increasing step. A step size smaller than epsilon lets PGD
search rather than jump straight to the edge.

```python
lower = original - epsilon
upper = original + epsilon
```

Defines the allowed L∞ box around the untouched image.

```python
adversarial = maximum(minimum(adversarial, upper), lower)
```

Projects every PGD step back into that box. This is the “projected” part of
Projected Gradient Descent and makes FGSM and PGD use the same total budget.

## 04 · Read the comparison

Both attacks begin from the clean image and attack its original predicted class.
FGSM uses one step; PGD uses repeated smaller steps. The amplified panel shows
PGD's pixel changes around neutral gray—those colors are deliberately magnified,
not what the model actually receives.

> **Threat model** · This is an untargeted, white-box classroom attack. It needs
> model gradients and tries to leave the original class—not choose a particular
> wrong answer.
"""


APP_CSS = """
#app-version {
  opacity: .62;
  font-size: .78rem;
  margin-top: -.65rem;
}
#tutorial-open {
  max-width: 13rem;
  border-radius: 999px !important;
}
#tutorial-modal {
  position: fixed !important;
  z-index: 1000;
  top: 1.25rem;
  right: 1.25rem;
  bottom: 1.25rem;
  width: min(680px, calc(100vw - 2.5rem));
  padding: 0 !important;
  overflow-y: auto;
  border: 1px solid color-mix(in srgb, var(--border-color-primary) 75%, transparent);
  border-radius: 24px;
  background: var(--background-fill-primary);
  box-shadow:
    0 0 0 100vmax rgba(8, 12, 20, .46),
    0 28px 80px rgba(8, 12, 20, .32),
    0 4px 16px rgba(8, 12, 20, .12);
  scrollbar-width: thin;
  isolation: isolate;
}
#tutorial-modal > .form {
  gap: 0 !important;
}
#tutorial-close {
  position: sticky;
  z-index: 4;
  top: 1rem;
  width: auto !important;
  min-width: 0 !important;
  max-width: max-content;
  margin: 1rem 1rem -3.75rem auto;
  padding: .48rem .9rem !important;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 999px !important;
  background: var(--background-fill-primary) !important;
  box-shadow: 0 4px 18px rgba(8, 12, 20, .12);
}
#tutorial-content {
  padding: 3.25rem clamp(1.4rem, 4vw, 3.5rem) 3rem;
}
#tutorial-content .prose {
  max-width: 58ch;
  margin: 0 auto;
  color: var(--body-text-color);
  font-size: 1rem;
  line-height: 1.7;
}
#tutorial-content .tutorial-kicker {
  display: inline-flex;
  margin-bottom: .7rem;
  color: #7c3aed;
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .15em;
}
#tutorial-content h1 {
  margin: 0 0 .75rem;
  max-width: 15ch;
  font-size: clamp(2rem, 5vw, 3.15rem);
  line-height: 1.03;
  letter-spacing: -.045em;
}
#tutorial-content h1 + p {
  margin: 0 0 2.5rem;
  max-width: 49ch;
  color: var(--body-text-color-subdued);
  font-size: 1.08rem;
}
#tutorial-content h2 {
  margin: 2.7rem 0 1.1rem;
  padding-top: 1.4rem;
  border-top: 1px solid var(--border-color-primary);
  font-size: 1.25rem;
  letter-spacing: -.02em;
}
#tutorial-content pre {
  margin: 1rem 0 .6rem;
  padding: .9rem 1.05rem;
  overflow-x: auto;
  border: 1px solid color-mix(in srgb, #8b5cf6 25%, var(--border-color-primary));
  border-radius: 12px;
  background: color-mix(in srgb, #8b5cf6 7%, var(--background-fill-secondary));
  box-shadow: inset 3px 0 0 #8b5cf6;
}
#tutorial-content pre code {
  color: var(--body-text-color);
  font-size: .88rem;
  line-height: 1.55;
}
#tutorial-content pre + p {
  margin: 0 0 1.45rem;
  color: var(--body-text-color-subdued);
}
#tutorial-content blockquote {
  margin: 2.6rem 0 0;
  padding: 1.1rem 1.25rem;
  border: 0;
  border-radius: 14px;
  background: color-mix(in srgb, #8b5cf6 10%, var(--background-fill-secondary));
}
@media (max-width: 640px) {
  #tutorial-modal {
    inset: 0;
    width: 100vw;
    border: 0;
    border-radius: 0;
  }
  #tutorial-content { padding: 3.4rem 1.25rem 2rem; }
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Adversarial Image Demo", css=APP_CSS) as demo:
        gr.Markdown(
            "# Fooling an image classifier: FGSM → PGD\n"
            "Upload a photo and compare a one-step attack with an iterative one "
            "against pretrained ResNet-18. Use only images you are permitted to upload."
        )
        gr.Markdown("`PGD tutorial edition`", elem_id="app-version")
        tutorial_open = gr.Button(
            "Read the code tutorial  →", variant="secondary", elem_id="tutorial-open"
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            with gr.Column():
                epsilon = gr.Slider(
                    0.0, 0.10, value=0.02, step=0.005,
                    label="Attack strength (epsilon)",
                )
                with gr.Row():
                    pgd_steps = gr.Slider(
                        1, 40, value=10, step=1, label="PGD steps"
                    )
                    pgd_step_size = gr.Slider(
                        0.001, 0.025, value=0.005, step=0.001,
                        label="PGD step size",
                    )
                attack = gr.Button("Classify, run FGSM, then PGD", variant="primary")
                results = gr.Markdown("Results will appear here.")
        with gr.Row():
            fgsm_image = gr.Image(label="1. After FGSM")
            pgd_image = gr.Image(label="2. After PGD")
            perturbation = gr.Image(label="PGD perturbation (amplified)")

        attack.click(
            fn=run_demo,
            inputs=[input_image, epsilon, pgd_steps, pgd_step_size],
            outputs=[results, fgsm_image, pgd_image, perturbation],
        )
        gr.Markdown(
            "**Teaching note:** This is an untargeted white-box attack. It assumes "
            "access to model gradients and pushes pixels away from the model's initial prediction."
        )

        with gr.Column(visible=False, elem_id="tutorial-modal") as tutorial_modal:
            tutorial_close = gr.Button("Close  ×", elem_id="tutorial-close")
            gr.Markdown(TUTORIAL, elem_id="tutorial-content")

        tutorial_open.click(
            lambda: gr.update(visible=True), outputs=tutorial_modal, queue=False
        )
        tutorial_close.click(
            lambda: gr.update(visible=False), outputs=tutorial_modal, queue=False
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
