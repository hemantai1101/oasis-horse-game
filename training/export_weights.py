"""
export_weights.py — Convert the trained PyTorch model to a browser-loadable JSON file.

Reads:   training/models/model.pt
Writes:  public/model/weights.json

The JSON format is a plain dict of layer weights:
  {
    "W1": [[...128 rows of 21 values...]],
    "b1": [...128 values...],
    "W2": [[...64 rows of 128 values...]],
    "b2": [...64 values...],
    "W3": [[...1 row of 64 values...]],
    "b3": [...1 value...]
  }

The browser's nnEvaluate() in ai.js reads this file and performs the forward pass
in pure JavaScript with no ML library.

Usage:
  python export_weights.py
  python export_weights.py --model models/model.pt --output ../public/model/weights.json
"""

import argparse
import json
from pathlib import Path

import torch

from train import ValueNet


def export_weights(model_path, output_path):
    model = ValueNet()
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Extract the 3 linear layers from the Sequential
    # model.net: [Linear(21→128), ReLU, Linear(128→64), ReLU, Linear(64→1), Tanh]
    linear_layers = [m for m in model.net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3, f'Expected 3 linear layers, got {len(linear_layers)}'

    weights = {}
    for i, layer in enumerate(linear_layers, start=1):
        # W shape: (out_features, in_features) — transpose for JS row-major matmul
        W = layer.weight.detach().numpy().tolist()   # [out][in]
        b = layer.bias.detach().numpy().tolist()     # [out]
        weights[f'W{i}'] = W
        weights[f'b{i}'] = b

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(weights, f, separators=(',', ':'))  # compact JSON

    size_kb = output_path.stat().st_size / 1024
    print(f'Exported to: {output_path}  ({size_kb:.1f} KB)')
    print('\nLayer shapes:')
    for i, layer in enumerate(linear_layers, start=1):
        print(f'  W{i}: {layer.weight.shape}  b{i}: {layer.bias.shape}')
    print('\nNext step: git add public/model/weights.json && git commit && git push')


def main():
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description='Export trained model weights to JSON')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model.pt (default: models/model.pt)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: ../public/model/weights.json)')
    args = parser.parse_args()

    model_path  = Path(args.model)  if args.model  else script_dir / 'models' / 'model.pt'
    output_path = Path(args.output) if args.output else script_dir.parent / 'public' / 'model' / 'weights.json'

    if not model_path.exists():
        print(f'ERROR: {model_path} not found.')
        print('Run:  python train.py  first.')
        return

    export_weights(model_path, output_path)


if __name__ == '__main__':
    main()
