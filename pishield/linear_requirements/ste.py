import torch
from torch.autograd import Function
from pishield.linear_requirements.shield_layer import ShieldLayer

class STEShield(Function):
    @staticmethod
    def forward(_, predictions: torch.Tensor, shield_layer: ShieldLayer) -> torch.Tensor:
        with torch.no_grad():
            shielded_predictions = shield_layer(predictions)
        return shielded_predictions
    
    @staticmethod
    def backward(_, grad_output: torch.Tensor) -> torch.Tensor:
        gradient = grad_output.clone()
        return gradient, None
