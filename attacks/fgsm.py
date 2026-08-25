"""Fast Gradient Sign Method (FGSM), written for readability."""

from collections.abc import Callable

import torch


def fgsm_attack(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    label: int,
    epsilon: float,
    logits_for: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take one loss-maximizing step bounded by ``epsilon`` per pixel channel."""
    attacked = pixels.detach().clone().requires_grad_(True)
    target = torch.tensor([label], device=pixels.device)
    loss = torch.nn.functional.cross_entropy(logits_for(model, attacked), target)

    model.zero_grad(set_to_none=True)
    loss.backward()
    gradient = attacked.grad.detach()

    adversarial = attacked + epsilon * gradient.sign()
    adversarial = adversarial.clamp(0, 1).detach()
    return adversarial, gradient
