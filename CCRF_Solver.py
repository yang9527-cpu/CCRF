from itertools import combinations
import numpy as np
import torch
from torch import nn
from CCRF_update import association, source_information
from torch.nn import functional as F
METHODS = ('CCRF', 'Top1', 'Top1Conf', 'FullLinear', 'NoSourceMIHalf')

def validate_probabilities(p):
    p = np.asarray(p)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1] < 3 or (p.shape[2] < 2):
        raise ValueError('Expected a nonempty (images, sources >= 3, classes >= 2) array')
    if not np.issubdtype(p.dtype, np.floating):
        raise ValueError('Expected floating-point probabilities')
    if not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    if not np.allclose(p.sum(axis=2, dtype=np.float64), 1.0, atol=0.0002, rtol=0):
        raise ValueError('Each image and source probability vector must sum to one')
    return p

def partitions(m):
    if m < 3:
        raise ValueError('At least three sources are required')
    return [part for part in combinations(range(m), m // 2) if m % 2 or 0 in part]

class Calibration(nn.Module):

    def __init__(self, m, k, linear=False, h=64):
        super().__init__()
        self.logtemp = nn.Parameter(torch.zeros(m, 1))
        self.bias = nn.Parameter(torch.zeros(m, k))
        self.linear = linear
        if linear:
            self.weight = nn.Parameter(torch.zeros(m, k, k))
        else:
            self.inw = nn.Parameter(torch.empty(m, k, h))
            nn.init.xavier_uniform_(self.inw)
            self.inb = nn.Parameter(torch.zeros(m, h))
            self.outw = nn.Parameter(torch.zeros(m, h, k))
            self.outb = nn.Parameter(torch.zeros(m, k))

    def forward(self, p):
        logs = p.clamp_min(1e-09).log()
        z = logs * torch.exp(self.logtemp)[None] + self.bias[None]
        if self.linear:
            return (z + torch.einsum('bmk,mkl->bml', logs, self.weight)).softmax(-1)
        h = F.gelu(torch.einsum('bmk,mkh->bmh', logs, self.inw) + self.inb)
        h = F.layer_norm(h, (h.shape[-1],))
        return (z + torch.einsum('bmh,mhk->bmk', h, self.outw) + self.outb).softmax(-1)

class Gate(nn.Module):

    def __init__(self, m, k):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(m, k))

    def forward(self, q, mask):
        logits = self.b[None].expand(len(q), -1, self.b.shape[-1])
        logits = logits.masked_fill(~mask[None, :, None].bool(), -1000000000.0)
        p = (logits.softmax(1) * q).sum(1)
        return p / p.sum(-1, keepdim=True)

class Fusion(nn.Module):

    def __init__(self, m, k, linear=False, *, method=None):
        super().__init__()
        self.method = method if method is not None else 'FullLinear' if linear else 'CCRF'
        if self.method not in METHODS or (linear and self.method != 'FullLinear'):
            raise ValueError('Unsupported model configuration')
        self.m, self.k = (m, k)
        self.cal = Calibration(m, k, self.method == 'FullLinear')
        self.gate = Gate(m, k)

    def forward(self, p):
        return self.gate(self.cal(p), torch.ones(p.shape[1], device=p.device))

def representation(p, method):
    if method not in {'Top1', 'Top1Conf'}:
        return p
    k = p.shape[-1]
    best = np.argmax(p, axis=-1, keepdims=True)
    if method == 'Top1':
        compact = np.zeros_like(p)
        np.put_along_axis(compact, best, 1.0, axis=-1)
    else:
        confidence = np.take_along_axis(p, best, axis=-1)
        compact = np.broadcast_to((1 - confidence) / (k - 1), p.shape).copy()
        np.put_along_axis(compact, best, confidence, axis=-1)
    return compact

def fit(p, seed=2026, method='CCRF', epochs=80, batch=1024, device='auto', gamma=1.0, after_epoch=None):
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if epochs < 1 or batch < 1 or (not np.isfinite(gamma)) or (gamma < 0):
        raise ValueError('Expected positive epochs and batch and nonnegative finite gamma')
    p = representation(validate_probabilities(p), method)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1993)
    n, m, k = p.shape
    model = Fusion(m, k, method=method).to(device)
    cal = model.cal
    residual = [cal.weight] if cal.linear else [cal.inw, cal.inb, cal.outw, cal.outb]
    groups = [dict(params=[cal.logtemp, cal.bias], lr=0.04, weight_decay=0.0), dict(params=residual, lr=0.003, weight_decay=0.001), dict(params=list(model.gate.parameters()), lr=0.01, weight_decay=0.001)]
    opt = torch.optim.AdamW(groups, foreach=False)
    x = torch.tensor(p, dtype=torch.float32, device=device)
    masks = []
    for part in partitions(m):
        a = torch.zeros(m, device=device)
        a[list(part)] = 1
        masks.append((a, 1 - a))
    pairs = torch.triu_indices(m, m, offset=1)
    for epoch in range(epochs):
        for ids in np.array_split(rng.permutation(n), max(1, int(np.ceil(n / batch)))):
            q = cal(x[ids])
            a, b = masks[int(rng.integers(len(masks)))]
            assoc = association(model.gate(q, a), model.gate(q, b))
            if method == 'NoSourceMIHalf' or gamma == 0:
                loss = -0.5 * assoc
            else:
                info = source_information(q, pairs)
                loss = -(assoc + info) / 2 if gamma == 1.0 else -0.5 * assoc - 0.5 * gamma * info
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, foreach=False)
            opt.step()
        if after_epoch is not None:
            after_epoch(epoch + 1, model)
    if not np.isfinite(float(loss.detach())):
        raise ValueError('Nonfinite training objective')
    return model.eval()

@torch.no_grad()
def predict(model, p, batch=1024):
    p = validate_probabilities(p)
    if p.shape[1:] != (model.m, model.k) or batch < 1:
        raise ValueError('Input axes must match fitted axes and batch must be positive')
    p = representation(p, model.method)
    device = next(model.parameters()).device
    return np.concatenate([model(torch.tensor(p[i:i + batch], dtype=torch.float32, device=device)).cpu().numpy() for i in range(0, len(p), batch)])
