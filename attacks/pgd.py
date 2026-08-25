"""Projected Gradient Descent (PGD), written for readability."""

from collections.abc import Callable

import torch


def pgd_attack(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    label: int,
    epsilon: float,
    step_size: float,
    steps: int,
    logits_for: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Take repeated signed-gradient steps inside an L-infinity pixel budget."""
    original = pixels.detach()
    adversarial = original.clone()
    target = torch.tensor([label], device=pixels.device)

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(
            logits_for(model, adversarial), target
        )
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + step_size * gradient.sign()
        lower = original - epsilon
        upper = original + epsilon
        adversarial = torch.maximum(torch.minimum(adversarial, upper), lower)
        adversarial = adversarial.clamp(0, 1)

    return adversarial.detach()
