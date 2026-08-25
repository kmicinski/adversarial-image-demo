"""Small, readable adversarial attacks used by the teaching demo."""

from .fgsm import fgsm_attack
from .pgd import pgd_attack

__all__ = ["fgsm_attack", "pgd_attack"]
