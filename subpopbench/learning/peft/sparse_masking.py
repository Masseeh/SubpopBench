import torch
from torch import nn
from torch.nn import functional as F

try:
    import spops
    from spops import csr_add, sddmm
except ImportError:
    pass

from torch.amp import custom_fwd
from subpopbench.utils.misc import find_module

def _build_random_csr(out_features: int, in_features: int, nnz: int, generator: torch.Generator, device):
    # Sample nnz unique positions directly on device
    N = out_features * in_features
    idx = torch.randperm(N, generator=generator, device=device)[:nnz]  # O(N) to generate perm, but avoids dense mask + nonzero
    rows = torch.div(idx, in_features, rounding_mode="floor")
    cols = idx - rows * in_features

    # Sort by (row, col) so CSR is contiguous per row
    sort_key = rows * in_features + cols
    perm = torch.argsort(sort_key)
    rows = rows[perm]
    cols = cols[perm]

    # Row counts and offsets
    row_counts = torch.bincount(rows, minlength=out_features)
    row_offsets = torch.empty(out_features + 1, dtype=torch.int32, device=device)
    row_offsets[0] = 0
    row_offsets[1:] = torch.cumsum(row_counts, dim=0, dtype=torch.int32)

    # Sputnik likes rows re-ordered by length (optional but keeps parity with your code)
    row_order = torch.argsort(row_counts, descending=True)

    return row_offsets, row_order.to(torch.int16), cols.to(torch.int16)

class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16, device_type='cuda')
    def forward(ctx, values, row_offsets, row_indices, col_indices, dense_weight, bias, input):
        # Save tensors for backward pass
        ctx.save_for_backward(values, row_offsets, row_indices, col_indices, dense_weight, bias, input)

        # Perform the sparse addition
        total_weight = csr_add(values, row_offsets, row_indices, col_indices, dense_weight)

        # Perform the linear operation
        output = F.linear(input, total_weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        values, row_offsets, row_indices, col_indices, dense_weight, bias, input = ctx.saved_tensors
        input_shape = input.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        input = input.reshape(-1, input.shape[-1])

        grad_values = sddmm(row_offsets, row_indices, col_indices, grad_output.T.contiguous(), input.T.contiguous(), backend="sputnik")

        grad_input = (grad_output @ csr_add(values, row_offsets, row_indices, col_indices, dense_weight)).reshape(
            input_shape)

        return grad_values, None, None, None, None, None, grad_input, None

class RandomMaskingLinear(nn.Module):
    def __init__(self, base_Linear: nn.Linear, masking_prob: float = 0.9, generator=None):
        super().__init__()
        self.base_Linear = base_Linear
        base_weight = self.base_Linear.weight

        self.out_features, self.in_features = base_weight.shape
        self.num_params   = self.out_features * self.in_features
        self.nnz          = int(self.num_params * (1-masking_prob))

        device = base_weight.device
        dtype  = base_weight.dtype

        self.generator = generator

        # Parameter for the sparse values (fixed size = nnz)
        self.tunable_weights = nn.Parameter(torch.zeros(self.nnz, dtype=dtype, device=device))

        # Fixed-size buffers; updated in-place on refresh
        self.register_buffer('row_offsets', torch.empty(self.out_features + 1, dtype=torch.int32, device=device))
        self.register_buffer('col_indices', torch.empty(self.nnz, dtype=torch.int16, device=device))
        self.register_buffer('row_indices', torch.empty(self.out_features, dtype=torch.int16, device=device))

        # Initial build of CSR pattern
        with torch.no_grad():
            ro, ri, ci = _build_random_csr(self.out_features, self.in_features, self.nnz, self.generator, device)
            self.row_offsets.copy_(ro)
            self.row_indices.copy_(ri)
            self.col_indices.copy_(ci)

        self.register_buffer("step", torch.tensor(1, dtype=torch.long))

        if self.generator is not None: self.generator = torch.Generator('cuda').manual_seed(self.generator.initial_seed())

    def forward(self, x):
        return SparseLinearFunction.apply(
            self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices,
            self.base_Linear.weight, self.base_Linear.bias, x
        )

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

class GMixoutLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.bfloat16, device_type='cuda')
    def forward(ctx, values, row_offsets, row_indices, col_indices, ratio, dense_weight, bias, input):
        # Save tensors & scalar needed for backward
        ctx.save_for_backward(values, row_offsets, row_indices, col_indices, dense_weight, bias, input)
        ctx.inv_r = 1.0 / float(ratio)  # ok to store as python float

        # === sparse-only adjustment: dense_weight + scatter(values / ratio) ===
        total_weight = csr_add(values * ctx.inv_r, row_offsets, row_indices, col_indices, dense_weight)

        return F.linear(input, total_weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        values, row_offsets, row_indices, col_indices, dense_weight, bias, input = ctx.saved_tensors
        in_shape = input.shape
        go = grad_output.reshape(-1, grad_output.shape[-1])
        x  = input.reshape(-1, input.shape[-1])

        # Gradient w.r.t. the CSR values corresponds to the scaled values (values/ratio)
        grad_values = sddmm(row_offsets, row_indices, col_indices,
                            go.T.contiguous(), x.T.contiguous(), backend="sputnik")
        # Chain rule for values_scaled = values * inv_r
        grad_values = grad_values * ctx.inv_r

        # grad_input needs the same effective weight used in forward:
        eff_weight = csr_add(values * ctx.inv_r, row_offsets, row_indices, col_indices, dense_weight)
        grad_input = (go @ eff_weight).reshape(in_shape)

        # Return grads for (values, row_offsets, row_indices, col_indices, ratio, dense_weight, bias, input)
        return grad_values, None, None, None, None, None, None, grad_input, None


class GMixoutLinear(nn.Module):
    def __init__(self, base_Linear: torch.Tensor, masking_prob: float = 0.9, mask_refresh: int = 1, mask_ema : float = 0.0,  keep_momentum: bool = False, use_mixout: bool = False,
                 generator=None):
        super().__init__()
        self.base_Linear = base_Linear
        self.ratio = (1-masking_prob)
        self.refresh_every = mask_refresh
        self.ema = mask_ema

        base_weight = self.base_Linear.weight

        self.out_features, self.in_features = base_weight.shape
        self.num_params   = self.out_features * self.in_features
        self.nnz          = int(self.num_params * self.ratio)

        device = base_weight.device
        dtype  = base_weight.dtype

        self.generator = generator

        # Parameter for the sparse values (fixed size = nnz)
        self.tunable_weights = nn.Parameter(torch.zeros(self.nnz, dtype=dtype, device=device))

        # Fixed-size buffers; updated in-place on refresh
        self.register_buffer('row_offsets', torch.empty(self.out_features + 1, dtype=torch.int32, device=device))
        self.register_buffer('col_indices', torch.empty(self.nnz, dtype=torch.int16, device=device))
        self.register_buffer('row_indices', torch.empty(self.out_features, dtype=torch.int16, device=device))

        # Initial build of CSR pattern
        with torch.no_grad():
            ro, ri, ci = _build_random_csr(self.out_features, self.in_features, self.nnz, self.generator, device)
            self.row_offsets.copy_(ro)
            self.row_indices.copy_(ri)
            self.col_indices.copy_(ci)

        self.register_buffer("step", torch.tensor(1, dtype=torch.long))

        if self.generator is not None: self.generator = torch.Generator('cuda').manual_seed(self.generator.initial_seed())

    @torch.no_grad()
    def refresh(self, base_weight):
        # Fast, GPU-only, in-place update of CSR; no dense mask, no reallocation
        total_weight = csr_add(self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices, base_weight)
        base_weight.data.copy_(base_weight.data * self.ema + total_weight.data * (1 - self.ema))

        ro, ri, ci = _build_random_csr(self.out_features, self.in_features, self.nnz, self.generator, base_weight.device)
        self.row_offsets.copy_(ro)
        self.row_indices.copy_(ri)
        self.col_indices.copy_(ci)
        self.tunable_weights.zero_()  # keep shape; cheap

    def forward(self, x):
        if self.training:
            if (self.step % self.refresh_every) == 0:
                self.refresh(self.base_Linear.weight)
            self.step += 1
            return GMixoutLinearFunction.apply(
                self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices, self.ratio,
                self.base_Linear.weight, self.base_Linear.bias, x
            )
        else:
            return F.linear(x, self.base_Linear.weight, self.base_Linear.bias)

class GMixout:
    def __init__(self, model, masking_prob, mask_refresh: int = 1, mask_ema : float = 0.0, keep_momentum: bool = False, use_mixout: bool = False, generator=None):
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
            if "tunable" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True