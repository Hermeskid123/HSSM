Milestone 1 | Can the matcher choose the correct index-finger path?
=================================================================

Render one hand at fixed position, scale and orientation. Keep the thumb and
other fingers fixed. Alternate between an extended index finger and an index
folded out of view. Save a binary silhouette, two finger-base endpoints and
the ground-truth configuration per example. Vary width, height and boundary
shape within each class. Use distinct seeds for reference and held-out hands.

Extract the external contour with OpenCV and select the index-side arc
between the endpoints. Resolve the two possible arcs by hand geometry, using
the same rule for both poses. Preserve point order. Build one reference arc
per configuration from the reference split.

First scoring rule (project adaptation)
Resample both reference arcs and the observed arc to the same number of
equally spaced arc-length samples. Register the endpoints to a common frame
and retain the finger height relative to the base width. Score each ordered
reference by mean squared Euclidean distance to the observed samples. Lower
is better. Equal sampling makes the two costs comparable. This baseline
checks the contour and scoring pipeline before implementing HSSM inference.

Verification
* A reference contour should score approximately zero against itself.
* Reversing contour traversal in the input should preserve the prediction
  after endpoint ordering is restored.
* Evaluate both classes on held-out shapes. Record sample ID, true label,
  prediction and both costs in CSV. Report the confusion matrix and accuracy.
* Save an overlay for each class and one for a failure if any occur. Inspect
  the selected arc and endpoints as well as the label.

Implemented CPU diagnostic
--------------------------
Linux / Python 3.12, from the repository root:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    PYTHONPATH=src .venv/bin/python -m pytest -q
    PYTHONPATH=src .venv/bin/python -m hssm.milestone1 --per-class 100 --output outputs/milestone1

The renderer, contour extraction, predictor and evaluator are independent
functions in src/hssm/milestone1.py. Predictor inputs exclude evaluation labels.
Images use (x,y) pixels; NumPy arrays use [row,column]. No angle parameters.
The palm and three non-index fingers and thumb stay fixed. Index base width,
height and asymmetric boundary shape vary; known endpoints vary with width.
Both poses use the same corridor rule: choose the endpoint-to-endpoint arc
with least mean squared horizontal excursion outside the endpoint corridor.
Nearest-pixel ties use coordinate ordering, independent of traversal direction.
128 equal arc-length samples are translated, rotated into endpoint coordinates
and divided by endpoint distance, retaining height relative to base width.
Scores are mean squared Euclidean distances; lower is better. Exact ties
choose extended by reference insertion order.

Reference seeds: extended 11, folded 12. Default held-out seeds: extended
10000..10099; folded 110000..110099. The fixed-view, clean synthetic classes
are deliberately well separated. This diagnostic is a project adaptation of
alternative hand structure (Wang et al., 2008, Figure 3), not the probabilistic
HSSM or registration in Sections 4-6, nor the real-image protocol in Section 7.
No claims about clutter, unknown endpoints, real images or intermediate poses.

Outputs include binary silhouettes for every example, reference silhouettes,
normalized references.npz, scores.csv (IDs, seeds, split, endpoints, labels,
predictions, both costs), summary.json and per-class overlays. Blue shows the
selected ground-truth synthetic index boundary; red shows the winning reference
mapped to its endpoints; green marks known endpoints. A failure overlay is
saved on the first misclassification, if any. Rerun in a new output directory
when changing evaluation size to avoid retaining older image files.

Validated default run: 200 held-out examples, accuracy 1.0, confusion matrix
(rows true, columns predicted, extended then folded): [[100,0],[0,100]].
Seven regression tests passed, including reference self-cost, traversal/start
invariance, normalized height preservation, deterministic rendering and saved
held-out evidence. Both overlays were visually inspected: the selected arc
stays on the index boundary and excludes the palm and neighboring fingers.
No misclassifications occurred; the extended reference is visibly shorter
than one held-out extended finger, as expected from shape variation.
