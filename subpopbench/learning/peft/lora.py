import torch
from torch import nn
from torch.nn import functional as F
import math
from subpopbench.utils.misc import find_module

class LoRALinear(torch.nn.Module):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            base_Linear: nn.Linear,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        super().__init__()

        self.base_Linear = base_Linear
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.base_Linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.base_Linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.base_Linear.weight.requires_grad = False
            if self.base_Linear.bias is not None:
                self.base_Linear.bias.requires_grad = False

            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        if fan_in_fan_out:
            self.base_Linear.weight.data = self.base_Linear.weight.data.transpose(0, 1)

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        super().train(mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.base_Linear.weight), bias=self.base_Linear.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                  1)) * self.scaling
        return result


class LoRA:

    def __init__(self, model, r):
        # find all of the Linear layer in model
        linear_layers = []
        for key, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((key, module))

        # replace all Linear layers with LoRA layers
        for key, module in linear_layers:
            parent_module, sub_key, _ = find_module(model, key)
            setattr(parent_module, sub_key,
                    LoRALinear(base_Linear=module, in_features=module.in_features,
                               out_features=module.out_features, r=r))

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True