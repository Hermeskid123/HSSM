"""Shared ordered-contour sampling and headless visualization."""
from pathlib import Path
import cv2
import numpy as np


def resample_arc(arc, samples):
    arc = np.asarray(arc, dtype=float)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(arc, axis=0), axis=1))]
    if samples < 2 or distance[-1] == 0:
        raise ValueError('Need a nonzero arc and at least two samples')
    keep = np.r_[True, np.diff(distance) > 0]
    target = np.linspace(0, distance[-1], samples)
    return np.column_stack([np.interp(target, distance[keep], arc[keep, i]) for i in (0, 1)])


def write_overlay(mask, truth, predicted, lines, path):
    """Independent truth in blue; actual registered image features in red."""
    canvas = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(canvas, 0, 30+20*(len(lines)+2), 0, 480,
                                cv2.BORDER_CONSTANT, value=(245,245,245))
    for curves, color, thickness in ((truth, (255,0,0), 3), (predicted, (0,0,255), 1)):
        for curve in curves:
            cv2.polylines(canvas, [np.rint(curve).astype(np.int32)], False, color, thickness)
    labels = [('Blue: renderer ground truth', (255,0,0)),
              ('Red: registered candidate image edges', (0,0,255))]
    labels += [(line, (0,0,0)) for line in lines]
    for i, (line, color) in enumerate(labels):
        cv2.putText(canvas, line, (8, mask.shape[0]+22+20*i),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(path)
