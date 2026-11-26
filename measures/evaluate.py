import numpy as np
from utils.data import read_probabilities

def score(prediction, labels, sources=None):
    with np.load(prediction, allow_pickle=False) as archive:
        q = archive['Q'].astype(np.float64)
        ids = archive['sample_id'].astype(str)
        classes = archive['class_names'].astype(str)
        pred = archive['pred'].astype(int)
    with np.load(labels, allow_pickle=False) as archive:
        y = archive['Y'].reshape(-1)
        label_ids = archive['sample_id'].astype(str)
        if 'class_names' in archive and (not np.array_equal(archive['class_names'].astype(str), classes)):
            raise ValueError('Reference class order differs from prediction.')
    if not np.array_equal(ids, label_ids):
        raise ValueError('Reference and prediction sample IDs differ or are in a different order.')
    n, k = (len(y), len(classes))
    if q.shape != (n, k) or not np.isfinite(q).all() or q.min() < 0 or (not np.allclose(q.sum(1), 1, atol=0.0002, rtol=0)):
        raise ValueError('Invalid saved probability matrix.')
    if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= k)):
        raise ValueError('Labels must be integer indices in the fixed class vocabulary.')
    if pred.shape != (n,) or not np.array_equal(pred, q.argmax(1)):
        raise ValueError('Saved winning classes disagree with Q.')
    cm = np.bincount(k * y + pred, minlength=k * k).reshape(k, k)
    hit = q[np.arange(n), y]
    result = dict(n=n, acc=100 * float((pred == y).mean()), macro_f1=200 * float(np.mean(np.diag(cm) / np.maximum(1, cm.sum(0) + cm.sum(1)))), brier100=100 * float(np.mean((q * q).sum(1) - 2 * hit + 1)), nll=float(-np.log(np.maximum(hit, 1e-12)).mean()))
    if sources is not None:
        p, source_ids, source_classes, _ = read_probabilities(sources)
        if not np.array_equal(source_ids, ids) or not np.array_equal(source_classes, classes):
            raise ValueError('Source identities/classes differ from the scored predictions.')
        covered = (p.argmax(2) == y[:, None]).any(1)
        good = pred == y
        u = int(covered.sum())
        r = int((~covered & good).sum())
        d = int((covered & ~good).sum())
        assert u + r - d == int(good.sum())
        result.update(covered=u, recovered=r, damaged=d, recovery=100 * r / (n - u) if n - u else None, damage=100 * d / u if u else None)
    return result
