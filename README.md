# Adversarial Image Classifier Demo

A small teaching demo for a cybersecurity + AI course. It uses a pretrained
ImageNet ResNet-18, classifies an uploaded image, and compares the Fast Gradient
Sign Method (FGSM) with iterative Projected Gradient Descent (PGD).

The interface shows both attacked predictions and images, an amplified PGD
perturbation, and an in-app code tutorial that opens as a modal alongside the demo.

## Run in Google Colab

Open `adversarial_image_demo.ipynb` in Colab and run its cells from top to
bottom. The first run downloads the public pretrained weights (about 45 MB).
Colab will print a public Gradio link near the final cell.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local URL printed by Gradio.

## What students should notice

- FGSM uses the sign of the gradient of classification loss with respect to the
  input pixels: `x_adv = clip(x + epsilon * sign(gradient), 0, 1)`.
- PGD repeats smaller signed-gradient steps and projects after every step so the
  result remains within the same epsilon budget around the original image.
- `epsilon` bounds the maximum per-channel pixel change. `0.02` is roughly
  `5/255`; the attack becomes more visible as epsilon increases.
- This is a white-box, untargeted attack: it has gradient access and tries to
  leave the original class, rather than force a specific target class.
- An attack is not guaranteed to flip every image at every epsilon. That is a
  useful basis for discussing robustness, attack budgets, and threat models.

## Responsible use

This intentionally limited classroom example attacks only a local, public image
classifier. Do not upload private or sensitive images to a shared Colab/Gradio
session, and do not use adversarial techniques to evade real-world safety or
security systems.
