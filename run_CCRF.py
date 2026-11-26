import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import CCRF_Solver as core
__version__ = '0.1.1'
from utils.data import read_probabilities, digest, write_json
from measures.evaluate import score
METHODS = list(core.METHODS)

def device(value):
    selected = 'cuda' if torch.cuda.is_available() else 'cpu'
    return selected if value == 'auto' else value

def fit(args):
    p, ids, classes, sources = read_probabilities(args.input)
    chosen = device(args.device)
    config = dict(method=args.method, epochs=args.epochs, batch=args.batch, gamma=args.gamma)
    model = core.fit(p, seed=args.seed, device=chosen, **config)
    target = Path(args.checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=1, version=__version__, config=config, seed=args.seed, source_names=sources.tolist(), class_names=classes.tolist(), fit_sample_ids=ids.tolist(), input_sha256=digest(args.input), state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()}), target)
    write_json(target.with_suffix('.json'), dict(schema=1, config=config, seed=args.seed, source_names=sources.tolist(), class_names=classes.tolist(), fit_images=len(ids), input_sha256=digest(args.input), checkpoint_sha256=digest(target), torch_version=str(torch.__version__), numpy_version=str(np.__version__), device=chosen, reference_labels_passed_to_fit=False))
    print(f'Saved fitted model: {target}')

def predict(args):
    p, ids, classes, sources = read_probabilities(args.input)
    chosen = device(args.device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if saved['schema'] != 1 or classes.tolist() != saved['class_names'] or sources.tolist() != saved['source_names']:
        raise ValueError('Checkpoint and prediction class/source identities or ordering differ.')
    config = saved['config']
    model = core.Fusion(len(sources), len(classes), method=config['method']).to(chosen)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    q = core.predict(model, p, batch=args.batch)
    if q.shape != (len(ids), len(classes)) or not np.isfinite(q).all():
        raise ValueError('Invalid predicted probabilities.')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, Q=q, pred=q.argmax(1), sample_id=ids, class_names=classes, source_names=sources)
    write_json(target.with_suffix('.json'), dict(schema=1, input_sha256=digest(args.input), checkpoint_sha256=digest(args.checkpoint), prediction_sha256=digest(target), n=len(ids), source_names=sources.tolist(), class_names=classes.tolist(), reference_labels_used=False))
    print(f'Saved predictions: {target}')

def evaluate(args):
    result = score(args.prediction, args.labels, args.sources)
    write_json(args.output, result)
    print(result)

def demo(args):
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    n, m, k = (96, 5, 4)
    y = rng.integers(k, size=n)
    logits = rng.normal(size=(n, m, k))
    logits[np.arange(n), :, y] += 1.5
    q = np.exp(logits - logits.max(2, keepdims=True))
    q /= q.sum(2, keepdims=True)
    ids = np.array([f'synthetic_{i:04d}' for i in range(n)])
    classes = np.array([f'class_{i}' for i in range(k)])
    sources = np.array([f'source_{i}' for i in range(m)])
    for name, selection in [('train', slice(0, 72)), ('test', slice(72, None))]:
        np.savez_compressed(folder / (name + '.npz'), P=q[selection], sample_id=ids[selection], class_names=classes, source_names=sources)
    np.savez_compressed(folder / 'labels.npz', Y=y[72:], sample_id=ids[72:], class_names=classes)
    fit(argparse.Namespace(input=folder / 'train.npz', checkpoint=folder / 'model.pt', device=args.device, method='CCRF', epochs=2, batch=32, gamma=1.0, seed=2026))
    predict(argparse.Namespace(input=folder / 'test.npz', checkpoint=folder / 'model.pt', device=args.device, output=folder / 'prediction.npz', batch=32))
    result = score(folder / 'prediction.npz', folder / 'labels.npz', folder / 'test.npz')
    write_json(folder / 'metrics.json', result)
    print('Synthetic installation check passed. These are not research benchmark results.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('fit')
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--method', choices=METHODS, default='CCRF')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.set_defaults(function=fit)
    p = commands.add_parser('predict')
    p.add_argument('--input', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.set_defaults(function=predict)
    p = commands.add_parser('evaluate')
    p.add_argument('--prediction', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--sources')
    p.add_argument('--output', required=True)
    p.set_defaults(function=evaluate)
    p = commands.add_parser('demo')
    p.add_argument('--output', default='demo_run')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu')
    p.set_defaults(function=demo)
    args = parser.parse_args()
    try:
        args.function(args)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')
if __name__ == '__main__':
    main()
