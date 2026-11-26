import torch

def association(a, b):
    same = (a * b).sum(-1).clamp_min(1e-12)
    marg_b = (a * b.mean(0, keepdim=True)).sum(-1).clamp_min(1e-12)
    marg_a = (a.mean(0, keepdim=True) * b).sum(-1).clamp_min(1e-12)
    return (0.5 * (2 * same.log() - marg_a.log() - marg_b.log())).mean()

def source_information(q, pairs):
    joint = torch.einsum('bmk,bnl->mnkl', q, q) / len(q)
    r = joint.sum(-1, keepdim=True)
    c = joint.sum(-2, keepdim=True)
    mi = (joint * (joint.clamp_min(1e-12).log() - (r * c).clamp_min(1e-12).log())).sum((-1, -2))
    return mi[pairs[0], pairs[1]].mean()
