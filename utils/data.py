from pathlib import Path
import hashlib
import json
import numpy as np

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')

def read_probabilities(path):
    with np.load(path, allow_pickle=False) as archive:
        p = archive['P']
        ids = archive['sample_id'].astype(str)
        classes = archive['class_names'].astype(str)
        source_key = next((key for key in ['source_names', 'model_ids', 'models'] if key in archive), None)
        if source_key is None:
            raise ValueError('Missing source_names (or model_ids/models).')
        sources = archive[source_key].astype(str)
    if p.ndim != 3 or p.shape[0] < 1 or p.shape[1] < 3 or (p.shape[2] < 2):
        raise ValueError('Expected nonempty probabilities P[images, sources>=3, classes>=2].')
    if not np.issubdtype(p.dtype, np.floating) or not np.isfinite(p).all() or p.min() < 0:
        raise ValueError('P must contain finite, nonnegative floating-point probabilities.')
    if not np.allclose(p.sum(2), 1.0, atol=0.0002, rtol=0):
        raise ValueError('Each source response must sum to one over classes.')
    for values, size, axis in [(ids, p.shape[0], 'sample_id'), (sources, p.shape[1], 'source_names'), (classes, p.shape[2], 'class_names')]:
        if values.shape != (size,) or len(set(values.tolist())) != size or any((not value for value in values)):
            raise ValueError(f'{axis} must contain one unique nonempty name per axis entry.')
    return (p, ids, classes, sources)
