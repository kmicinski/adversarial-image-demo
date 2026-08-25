"""A small Gradio demo comparing FGSM and PGD attacks on ImageNet images."""

from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from attacks import fgsm_attack, pgd_attack


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
    fgsm_image, _ = fgsm_attack(model, pixels, clean_id, epsilon, logits_for)
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
        logits_for,
    )
    with torch.no_grad():
        pgd_logits = logits_for(model, pgd_image)
    pgd_id, pgd_confidence = prediction(pgd_logits)

    # Amplify the tiny pixel differences so students can inspect their structure.
    delta = pgd_image - pixels
    amplified = (delta / (2 * max(epsilon, 1e-6)) + 0.5).clamp(0, 1)
    fgsm_status = "Changed" if clean_id != fgsm_id else "Unchanged"
    pgd_status = "Changed" if clean_id != pgd_id else "Unchanged"
    report = f"""
### Prediction comparison

| Stage | Top prediction | Confidence | Outcome |
| :-- | :-- | --: | :-- |
| Original | **{LABELS[clean_id]}** | {clean_confidence:.1%} | Baseline |
| FGSM | **{LABELS[fgsm_id]}** | {fgsm_confidence:.1%} | {fgsm_status} |
| PGD | **{LABELS[pgd_id]}** | {pgd_confidence:.1%} | {pgd_status} |

<span class="budget-chip">ε {epsilon:.3f} · {epsilon * 255:.1f}/255</span>
<span class="budget-chip">PGD {int(pgd_steps)} × {float(pgd_step_size):.3f}</span>
"""
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

[Open the complete FGSM source ↗](https://raw.githubusercontent.com/kmicinski/adversarial-image-demo/main/attacks/fgsm.py)

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

[Open the complete PGD source ↗](https://raw.githubusercontent.com/kmicinski/adversarial-image-demo/main/attacks/pgd.py)

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
.gradio-container {
  --solar-base3: #fdf6e3;
  --solar-base2: #eee8d5;
  --solar-base1: #93a1a1;
  --solar-base00: #657b83;
  --solar-base01: #586e75;
  --solar-blue: #268bd2;
  --solar-cyan: #2aa198;
  --background-fill-primary: #fdf6e3;
  --background-fill-secondary: #eee8d5;
  --block-background-fill: #fffaf0;
  --block-border-color: #ddd6c4;
  --border-color-primary: #ddd6c4;
  --body-text-color: #586e75;
  --body-text-color-subdued: #657b83;
  max-width: 1180px !important;
  width: 100% !important;
  margin-right: auto !important;
  margin-left: auto !important;
  padding: clamp(1rem, 3vw, 2.5rem) !important;
  background: var(--solar-base3);
  color: var(--solar-base01);
}
body {
  margin: 0;
  background: #fdf6e3;
}
#workspace-row, #results-panel, #outputs-heading, #output-row, #footer-note {
  width: 100%;
  margin-right: auto !important;
  margin-left: auto !important;
}
#hero {
  max-width: 760px;
  margin: 0 auto;
  padding: clamp(1.5rem, 4vw, 3.25rem) 1rem 1.5rem;
  text-align: center;
}
#hero .hero-kicker {
  margin-bottom: .9rem;
  color: var(--solar-base00);
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .11em;
  text-transform: uppercase;
}
#hero h1 {
  margin: 0;
  color: #073642;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2rem, 5vw, 3.15rem);
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: -.025em;
}
#hero p {
  max-width: 620px;
  margin: 1rem auto 0;
  color: var(--solar-base00);
  font-size: 1rem;
  line-height: 1.6;
}
#hero-actions {
  justify-content: center;
  align-items: center;
  max-width: 760px;
  margin: 0 auto 2rem;
}
#tutorial-open {
  min-width: 13.5rem;
  max-width: 13.5rem;
  border-color: #b9b29f !important;
  border-radius: 7px !important;
  background: #fffaf0 !important;
  color: #075985 !important;
}
#source-links {
  margin: 0 !important;
  color: var(--body-text-color-subdued);
  font-size: .86rem;
}
#source-links a, #tutorial-content a {
  color: var(--solar-blue);
  font-weight: 650;
  text-decoration: none;
}
#source-links a:hover, #tutorial-content a:hover { text-decoration: underline; }
#workspace-row { gap: 1.15rem; align-items: stretch; }
#input-image, #control-panel, #results-panel, .output-card {
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--border-color-primary) 82%, transparent) !important;
  border-radius: 9px !important;
  background: #fffaf0 !important;
  box-shadow: none;
}
#input-image { min-height: 390px; }
#control-panel { padding: clamp(1rem, 2vw, 1.4rem); }
#control-heading h3 { margin: 0; font-size: 1.1rem; }
#control-heading p {
  margin: .3rem 0 .5rem;
  color: var(--body-text-color-subdued);
  font-size: .88rem;
}
#run-attack {
  min-height: 3.1rem;
  margin-top: .35rem;
  border: 0 !important;
  border-radius: 7px !important;
  background: var(--solar-blue) !important;
  color: #fdf6e3 !important;
  font-weight: 750;
  box-shadow: none;
}
#results-panel {
  margin-top: 1.15rem;
  padding: 1rem 1.3rem;
}
#results-panel h3 { margin-top: .15rem; }
#results-panel table { margin: .65rem 0 1rem; }
#results-panel th { color: var(--body-text-color-subdued); font-size: .8rem; }
#results-panel td { padding-top: .75rem; padding-bottom: .75rem; }
#results-panel .budget-chip {
  display: inline-flex;
  margin: .15rem .35rem .1rem 0;
  padding: .32rem .65rem;
  border: 1px solid var(--border-color-primary);
  border-radius: 5px;
  background: var(--solar-base2);
  color: var(--body-text-color-subdued);
  font-size: .76rem;
  font-weight: 650;
}
#outputs-heading { margin: 2rem 0 .7rem; }
#outputs-heading h2 { margin-bottom: .25rem; font-size: 1.35rem; }
#outputs-heading p { color: var(--body-text-color-subdued); }
#output-row { gap: 1rem; }
.output-card { padding: .35rem; }
#footer-note {
  margin: 1.5rem 0 .5rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border-color-primary);
  color: var(--body-text-color-subdued);
  font-size: .82rem;
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
  border: 1px solid #d4cdbb;
  border-radius: 10px;
  background: var(--solar-base3);
  box-shadow:
    0 0 0 100vmax rgba(0, 43, 54, .28),
    0 18px 48px rgba(0, 43, 54, .18);
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
  border-radius: 6px !important;
  background: #fffaf0 !important;
  color: var(--solar-base01) !important;
  box-shadow: none;
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
  color: var(--solar-cyan);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .15em;
}
#tutorial-content h1 {
  margin: 0 0 .75rem;
  max-width: 15ch;
  color: #073642;
  font-family: Georgia, "Times New Roman", serif;
  font-size: clamp(2rem, 5vw, 2.8rem);
  line-height: 1.08;
  letter-spacing: -.025em;
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
  border: 1px solid #d4cdbb;
  border-radius: 6px;
  background: var(--solar-base2);
  box-shadow: inset 3px 0 0 var(--solar-cyan);
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
  border-radius: 6px;
  background: var(--solar-base2);
}
@media (max-width: 640px) {
  .gradio-container { padding: .85rem !important; }
  #hero h1 { font-size: 2.15rem; }
  #hero-actions { align-items: flex-start; }
  #tutorial-open { max-width: none; width: 100%; }
  #input-image { min-height: 310px; }
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
    theme = gr.themes.Base(
        primary_hue="blue", neutral_hue="slate", radius_size="sm"
    )
    with gr.Blocks(title="Adversarial Image Lab", theme=theme, css=APP_CSS) as demo:
        gr.HTML(
            """
            <section id="hero">
              <div class="hero-kicker">Cybersecurity + AI · Classroom Lab</div>
              <h1>Adversarial Examples: FGSM and PGD</h1>
              <p>Compare two white-box attacks on a pretrained ResNet-18 and
              examine how bounded pixel changes affect its predictions.</p>
            </section>
            """
        )
        with gr.Row(elem_id="hero-actions"):
            tutorial_open = gr.Button(
                "Read the code tutorial  →",
                variant="secondary",
                elem_id="tutorial-open",
            )
            gr.Markdown(
                "[Raw FGSM source ↗](https://raw.githubusercontent.com/kmicinski/adversarial-image-demo/main/attacks/fgsm.py)"
                " &nbsp;·&nbsp; "
                "[Raw PGD source ↗](https://raw.githubusercontent.com/kmicinski/adversarial-image-demo/main/attacks/pgd.py)",
                elem_id="source-links",
            )

        with gr.Row(elem_id="workspace-row"):
            input_image = gr.Image(
                type="pil",
                label="Input image",
                height=390,
                elem_id="input-image",
                scale=6,
            )
            with gr.Column(elem_id="control-panel", scale=5):
                gr.Markdown(
                    "### Attack settings\n"
                    "Both methods use the same maximum per-pixel budget.",
                    elem_id="control-heading",
                )
                epsilon = gr.Slider(
                    0.0, 0.10, value=0.02, step=0.005,
                    label="Pixel budget (epsilon)",
                    info="0.02 is about 5/255 per color channel.",
                )
                with gr.Row():
                    pgd_steps = gr.Slider(
                        1, 40, value=10, step=1, label="PGD iterations"
                    )
                    pgd_step_size = gr.Slider(
                        0.001, 0.025, value=0.005, step=0.001,
                        label="PGD step size",
                    )
                attack = gr.Button(
                    "Run both attacks  →", variant="primary", elem_id="run-attack"
                )

        results = gr.Markdown(
            "### Prediction comparison\nUpload an image and run both attacks to compare their predictions.",
            elem_id="results-panel",
        )

        gr.Markdown(
            "## Inspect the outputs\n"
            "PGD's perturbation is amplified around neutral gray so its structure is visible.",
            elem_id="outputs-heading",
        )
        with gr.Row(elem_id="output-row"):
            fgsm_image = gr.Image(
                label="01 · After FGSM", height=280, elem_classes="output-card"
            )
            pgd_image = gr.Image(
                label="02 · After PGD", height=280, elem_classes="output-card"
            )
            perturbation = gr.Image(
                label="03 · PGD perturbation × amplified",
                height=280,
                elem_classes="output-card",
            )

        attack.click(
            fn=run_demo,
            inputs=[input_image, epsilon, pgd_steps, pgd_step_size],
            outputs=[results, fgsm_image, pgd_image, perturbation],
        )
        gr.Markdown(
            "**For teaching and research.** This untargeted white-box demonstration "
            "attacks only the local, public model shown here. Avoid uploading private "
            "images to a shared session.",
            elem_id="footer-note",
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
