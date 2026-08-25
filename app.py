"""A small Gradio demo of an FGSM adversarial attack on ImageNet images."""

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


def to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    return np.uint8(np.clip(array * 255.0, 0, 255))


def run_demo(image: Image.Image | None, epsilon: float):
    if image is None:
        raise gr.Error("Please upload an image first.")

    model = get_model()
    pixels = image_to_tensor(image)

    with torch.no_grad():
        clean_logits = logits_for(model, pixels)
    clean_id, clean_confidence = prediction(clean_logits)

    adversarial, _ = fgsm_attack(model, pixels, clean_id, float(epsilon))
    with torch.no_grad():
        adversarial_logits = logits_for(model, adversarial)
    adversarial_id, adversarial_confidence = prediction(adversarial_logits)

    # Amplify the tiny pixel differences so students can inspect their structure.
    delta = adversarial - pixels
    amplified = (delta / (2 * max(float(epsilon), 1e-6)) + 0.5).clamp(0, 1)
    changed = "Yes" if clean_id != adversarial_id else "No"
    report = (
        f"### Results\n"
        f"- **Original:** {LABELS[clean_id]} ({clean_confidence:.1%})\n"
        f"- **After FGSM:** {LABELS[adversarial_id]} ({adversarial_confidence:.1%})\n"
        f"- **Prediction changed:** {changed}\n"
        f"- **Epsilon:** {epsilon:.3f}, or about {epsilon * 255:.1f}/255 per channel"
    )
    return report, to_numpy_image(adversarial), to_numpy_image(amplified)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Adversarial Image Demo") as demo:
        gr.Markdown(
            "# Fooling an image classifier with FGSM\n"
            "Upload a photo, then take one gradient-based step that tries to make "
            "a pretrained ResNet-18 wrong. Use only images you are permitted to upload."
        )
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            with gr.Column():
                epsilon = gr.Slider(
                    0.0, 0.10, value=0.02, step=0.005,
                    label="Attack strength (epsilon)",
                )
                attack = gr.Button("Classify and attack", variant="primary")
                results = gr.Markdown("Results will appear here.")
        with gr.Row():
            adversarial_image = gr.Image(label="Perturbed image")
            perturbation = gr.Image(label="Perturbation (amplified for visibility)")

        attack.click(
            fn=run_demo,
            inputs=[input_image, epsilon],
            outputs=[results, adversarial_image, perturbation],
        )
        gr.Markdown(
            "**Teaching note:** This is an untargeted white-box attack. It assumes "
            "access to model gradients and pushes pixels away from the model's initial prediction."
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
