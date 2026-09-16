"""Compare rendered figure files with the frozen manuscript artwork."""
import argparse
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


def pixels(path):
    if path.suffix.lower() == '.pdf':
        with fitz.open(path) as doc:
            if len(doc) != 1:
                raise ValueError(f'Expected single-page figure: {path}')
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            return Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
    return Image.open(path).convert('RGB')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    results = []
    for ref in sorted(a.reference.rglob('*')):
        if ref.suffix.lower() not in {'.pdf', '.png'}:
            continue
        rel = ref.relative_to(a.reference)
        other = a.candidate / rel
        if not other.is_file():
            results.append({'file': str(rel), 'missing': True})
            continue
        left, right = pixels(ref), pixels(other)
        shape_equal = left.size == right.size
        scaled = right.resize(left.size)
        difference = np.abs(np.asarray(left, dtype=float) - np.asarray(scaled, dtype=float))
        results.append({'file': str(rel), 'reference_size': left.size,
                        'candidate_size': right.size, 'same_size': shape_equal,
                        'mean_absolute_pixel_difference_255': float(difference.mean()),
                        'identical_rendering': shape_equal and not difference.any()})
        canvas = Image.new('RGB', (left.width * 2, left.height), 'white')
        canvas.paste(left, (0, 0))
        canvas.paste(scaled, (left.width, 0))
        canvas.thumbnail((1800, 1500))
        canvas.save(a.output / (ref.stem + '_comparison.png'))
    (a.output / 'render_comparison.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
