import torch
from torch import nn
from torch.nn import functional as F
from subpopbench.utils.misc import find_module

class RandomMaskingLinear(nn.Module):
    def __init__(self, base_Linear, masking_prob=0.9, generator=None):
        super().__init__()
        self.base_Linear = base_Linear
        weight = self.base_Linear.weight
        bias = self.base_Linear.bias

        out_dim, in_dim = weight.shape
        num_params = out_dim * in_dim + out_dim
        self.ratio = 1 - float(masking_prob)
        num_masked = int(num_params * self.ratio)

        # randomly select the optimized parameters
        masked_indexs = torch.randperm(num_params, generator=generator)[:num_masked]
        mask = torch.zeros(num_params, dtype=bool).scatter(dim=0, index=masked_indexs, value=True)
        mask = mask.reshape(out_dim, in_dim + 1)
        self.mask_weight = mask[:,:-1]
        self.mask_bias = mask[:,-1]

        self.tunable_weight = nn.Parameter(torch.masked_select(weight.detach(), mask=self.mask_weight))
        self.tunable_bias = nn.Parameter(torch.masked_select(bias.detach(), mask=self.mask_bias))

    def forward(self, x):
        self.mask_weight = self.mask_weight.to(self.tunable_weight.device)
        self.mask_bias = self.mask_bias.to(self.tunable_bias.device)

        if self.mask_weight.sum() > 0:
            weight = torch.masked_scatter(self.base_Linear.weight, mask=self.mask_weight, source=self.tunable_weight)
        else:
            weight = self.base_Linear.weight

        if self.mask_bias.sum() > 0:
            bias = torch.masked_scatter(self.base_Linear.bias, mask=self.mask_bias, source=self.tunable_bias)
        else:
            bias = self.base_Linear.bias

        return F.linear(x, weight, bias)

class RandomMasking:
    def __init__(self, model, masking_prob, generator):
        assert 0.0 <= masking_prob <= 1.0

        # find all of the Linear layer in model
        linear_layers = []
        for key, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((key, module))

        # replace all Linear layers with LoRA layers
        for key, module in linear_layers:
            parent_module, sub_key, _ = find_module(model, key)
            setattr(parent_module, sub_key,
                    RandomMaskingLinear(base_Linear=module, masking_prob=masking_prob, generator=generator))

        for n, p in model.named_parameters():
            if "tunable" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

class GMixoutLinear(nn.Module):
    def __init__(self,
                 base_Linear,
                 masking_prob        : float = 0.9,
                 mask_refresh : int   = 100,
                 mask_ema     : float = 0.0,
                 keep_momentum: bool = True,
                 use_mixout: bool = False,
                 generator=None):

        super().__init__()
        self.base_Linear = base_Linear
        base_weight = self.base_Linear.weight
        base_bias = self.base_Linear.bias

        self.out_dim, self.in_dim = base_weight.shape
        self.num_w = base_weight.numel()
        self.num_b = base_bias.numel()
        self.keep_momentum = keep_momentum
        self.use_mixout = use_mixout

        self.ratio = 1 - float(masking_prob)
        self.sel_w = int(self.num_w * self.ratio)
        self.sel_b = int(self.num_b * self.ratio)
        self.refresh = int(mask_refresh)
        self.ema = float(mask_ema)

        self.generator = generator

        # ---- index list for current subset (buffers) ---------------------
        self.register_buffer("step", torch.tensor(1, dtype=torch.long))
        dev = base_weight.device
        self.register_buffer("w_idx", torch.empty(self.sel_w, dtype=torch.long, device=dev))
        self.register_buffer("b_idx", torch.empty(self.sel_b, dtype=torch.long, device=dev))
        self._sample_subset(base_weight.device)                       # fills w_idx / b_idx

        # ---- trainable deltas (full size) --------------------------------
        self.delta_w = nn.Parameter(torch.zeros_like(base_weight))
        self.delta_w.data.copy_(base_weight.data)  # copy base values into deltas
        self.delta_b = nn.Parameter(torch.zeros_like(base_bias))
        self.delta_b.data.copy_(base_bias.data)  # copy base values into deltas

        # ---- single gradient-mask hook -----------------------------------
        self._hook_w = self.delta_w.register_hook(self._make_grad_hook(self.w_idx))
        self._hook_b = self.delta_b.register_hook(self._make_grad_hook(self.b_idx))

        self._optim = None                          # filled by link_optimizer()
        if self.generator is not None: self.generator = torch.Generator('cuda').manual_seed(self.generator.initial_seed())

    # ------------------------------------------------------------------ API
    def link_optimizer(self, optim: torch.optim.Optimizer):
        """Call exactly once *after* the optimiser is created."""
        self._optim = optim

    # ---------------------------------------------------------------- helpers
    @torch.no_grad()
    def _sample_subset(self, device):
        """Draw a new random subset (on current device)."""
        if self.sel_w:
            self.w_idx.copy_(torch.randperm(self.num_w, device=device, generator=self.generator)[: self.sel_w])
        if self.sel_b:
            self.b_idx.copy_(torch.randperm(self.num_b, device=device, generator=self.generator)[: self.sel_b])

    @staticmethod
    def _make_grad_hook(idx):
        """Return a closure that zeros grads outside idx."""
        if idx.numel() == 0:          # allow ratio == 0
            return lambda g: torch.zeros_like(g)

        idx = idx.clone()             # capture a *copy* that never changes

        def hook(grad):
            flat = grad.view(-1).clone()
            flat.zero_()              # faster than masked_fill for sparse set
            flat[idx] = grad.view(-1)[idx]
            return flat.view_as(grad)
        return hook

    @torch.no_grad()
    def _update_optimizer_state(self, param, o_idx, n_idx):
        if self._optim is None:
            return
        if param in self._optim.state:
            for k, v in self._optim.state[param].items():
                if k == "step":
                    continue
                if torch.is_tensor(v):
                    cp_v = v.view(-1).clone()
                    v.zero_()
                    v = v.view(-1)
                    v[n_idx] = cp_v[o_idx]
                    v = v.view_as(param)

    @torch.no_grad()
    def _merge_and_refresh(self, p_weight, p_bias, ema):
        """EMA-merge deltas, reset optimiser state, resample subset."""

        p_weight.data.copy_(p_weight.data * ema + self.delta_w.data * (1 - ema))
        p_bias.data.copy_(p_bias.data * ema + self.delta_b.data * (1 - ema))

        if not self.use_mixout:
            self.delta_w.data.copy_(p_weight.data)  # copy base values into deltas
            self.delta_b.data.copy_(p_bias.data)  # copy base values into deltas

        old_w_idx = self.w_idx.clone()
        old_b_idx = self.b_idx.clone()

        self._sample_subset(p_weight.device)

        if self.keep_momentum:
            self._update_optimizer_state(self.delta_w, old_w_idx, self.w_idx)
            self._update_optimizer_state(self.delta_b, old_b_idx, self.b_idx)

        self._hook_w.remove()
        self._hook_b.remove()
        self._hook_w = self.delta_w.register_hook(self._make_grad_hook(self.w_idx))
        self._hook_b = self.delta_b.register_hook(self._make_grad_hook(self.b_idx))

    # ---------------------------------------------------------------- forward
    def _stitch_weight(self, p_weight):
        # Build full weight as autograd-tracked tensor (cheap).
        flat = p_weight.view(-1).clone()
        if self.sel_w:
            flat.index_copy_(0, self.w_idx, self.delta_w.view(-1)[self.w_idx])
        return flat.view_as(p_weight)

    def _stitch_bias(self, p_bias):
        flat = p_bias.view(-1).clone()
        if self.sel_b:
            flat.index_copy_(0, self.b_idx, self.delta_b.view(-1)[self.b_idx])
        return flat.view_as(p_bias)

    def forward(self, x):
        p_weight = self.base_Linear.weight
        p_bias = self.base_Linear.bias

        if self.training:
            if self.step == 1:
                # First step: copy base values into deltas. Especially important for resuming training.
                self._merge_and_refresh(p_weight, p_bias, 1.0)
            if self.step % self.refresh == 0:
                self._merge_and_refresh(p_weight, p_bias, self.ema)

            self.step += 1

            w = self._stitch_weight(p_weight)
            b = self._stitch_bias(p_bias)

            if self.ratio > 0:
                inv_r = 1.0 / self.ratio

                w = (w - p_weight * (1-self.ratio)) * inv_r
                b = (b - p_bias * (1-self.ratio)) * inv_r

            return F.linear(x, w, b)
        else:
            if self.use_mixout:
                return F.linear(x, self.delta_w, self.delta_b)
            else:
                return F.linear(x, p_weight, p_bias)
            

class GMixout:
    def __init__(self, model, masking_prob, mask_refresh: int = 1, mask_ema : float = 0.0, keep_momentum: bool = True, use_mixout: bool = False, generator=None):
        assert 0.0 <= masking_prob <= 1.0

        # find all of the Linear layer in model
        linear_layers = []
        for key, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((key, module))

        # replace all Linear layers with LoRA layers
        for key, module in linear_layers:
            parent_module, sub_key, _ = find_module(model, key)
            setattr(parent_module, sub_key,
                    GMixoutLinear(base_Linear=module, masking_prob=masking_prob, mask_refresh=mask_refresh, mask_ema=mask_ema, keep_momentum=keep_momentum, use_mixout=use_mixout, generator=generator))

        for n, p in model.named_parameters():
            if "delta" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True