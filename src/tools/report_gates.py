"""Print every fusion gate in a checkpoint, to tell a dead branch from a useless one.

Each auxiliary branch enters its host as ``host + gate * residual`` with the gate
initialized at zero, so a branch that changed nothing has two very different
explanations and they are distinguishable only by the gate's trained value:

  gate still ~0   the branch never opened. Its parameters received gradient
                  proportional to the gate, so a zero gate means the experiment
                  did not actually run. Seeding a small positive gate retries it.

  gate clearly >0 the branch opened, the optimizer used it, and the result was
                  still flat. The information is redundant; adding features to
                  this branch will not help.

    python -m src.tools.report_gates --checkpoint runs/<run>/ckpt/best.pt
"""

import argparse
from pathlib import Path

import torch

# channel_gate: GatedFuse (forensic). gamma: wavelet/luma residuals. scale: DCT arms.
SUFFIXES = ('channel_gate', 'gamma', 'scale')


def gates(state: dict) -> list[tuple[str, float, float]]:
    found = []
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not key.endswith(SUFFIXES):
            continue
        magnitude = value.float().abs()
        found.append((key, float(magnitude.mean()), float(magnitude.max())))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--section', default='ema', choices=('model', 'ema'))
    parser.add_argument('--open-above', type=float, default=.01,
                        help='gates with a smaller mean magnitude are reported as never opened')
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = saved.get(args.section)
    if not isinstance(state, dict):
        raise SystemExit(f"checkpoint has no '{args.section}' state dict")

    found = gates(state)
    if not found:
        raise SystemExit('no fusion gates found in this checkpoint')
    width = max(len(key) for key, _, _ in found)
    for key, mean, peak in sorted(found):
        verdict = 'OPEN' if mean > args.open_above else 'never opened'
        print(f'{key:<{width}}  mean|g|={mean:.5f}  max|g|={peak:.5f}  {verdict}')
    print(f"\nepoch={saved.get('epoch')} best_aic={saved.get('best_aic')}")


if __name__ == '__main__':
    main()
