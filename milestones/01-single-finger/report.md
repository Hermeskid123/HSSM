# Milestone 1: choosing an index-finger contour

## Purpose and scientific question

Milestone 1 asks whether a simple contour matcher can distinguish an extended
index finger from a finger folded out of view, while the rest of the hand stays
fixed. It establishes a small, inspectable baseline before introducing the full
hand model or searching for a hand in an image.

The experiment checks the complete path from a rendered silhouette to a class
decision: extract a boundary, select the relevant arc, represent it consistently,
compare it with two references, and save evidence for evaluation. A successful
result shows that this controlled pipeline works on new synthetic shapes.

The motivation comes from the alternative hand structures in Figure 3 of Wang,
Athitsos, Sclaroff and Betke, *Detecting Objects of Variable Shape Structure with
Hidden State Shape Models* (2008), available in
[`references/wang_pami2008.pdf`](../../references/wang_pami2008.pdf).
Synthetic silhouettes, known finger-base endpoints, the corridor rule and the
distance score are **project adaptations**. This milestone does not implement
the paper's probabilistic formulation, registration or inference in Sections
4–6, and does not reproduce its Section 7 real-image protocol.

## How the code works

The implementation is in [`src/hssm/milestone1.py`](../../src/hssm/milestone1.py).
Rendering, prediction, evaluation and visualization are callable functions,
rather than operations coupled to a graphical application. Everything runs on
CPU using NumPy and headless OpenCV. Coordinates are `(x, y)` pixels; image
arrays use `[row, column]`, with image y increasing downward.

| Function | Responsibility |
|---|---|
| `render(label, seed)` | Generate a binary hand silhouette and known base endpoints. |
| `external_contour(mask)` | Extract the single connected external boundary. |
| `select_arc(contour, endpoints)` | Choose and order the index-side boundary. |
| `normalize(arc, samples=128)` | Resample and register an arc into an endpoint frame. |
| `predict(contour, endpoints, references)` | Compute both costs and choose the lower-cost class. |
| `overlay(...)` | Draw the observed arc, winning reference and endpoints. |
| `evaluate(output, per_class)` | Build references, score held-out examples and save results. |
| `main()` | Expose evaluation through the command line. |

### 1. Generate controlled silhouettes

`render` creates a 256 × 256 binary image. A fixed polygon supplies the palm,
thumb and other three fingers. An additional arch supplies the index boundary.
An extended example has a tall arch; a folded example has a shallow arch at the
finger base. The folded class is a silhouette abstraction, not an articulated
3D finger bending simulation.

The seeded generator varies the base width from 20 to 28 pixels, the nominal
extended height from 56 to 80 pixels, and the nominal folded height from 4 to
12 pixels. Small sinusoidal perturbations vary the boundary asymmetrically.
Position and the remaining hand geometry stay fixed. The renderer returns the
two base endpoints as deliberate known inputs; the matcher does not locate
them independently.

### 2. Extract and select the boundary

`external_contour` uses OpenCV's `RETR_EXTERNAL` and `CHAIN_APPROX_NONE`, retaining
the ordered boundary pixels. It rejects silhouettes with more or fewer than
one external contour.

A closed boundary has two paths between the base endpoints. `select_arc`
constructs both, ordered from the first endpoint toward the second. It chooses
the path with the smallest mean squared horizontal excursion outside the
endpoints' x corridor. The index arc remains within that corridor; the other
path travels around the palm and neighboring fingers. Both poses use this
same geometry rule, with no true class label supplied to it.

Rasterization can place an endpoint equally near multiple contour pixels.
The code resolves that tie by pixel coordinates, so changing the contour's
starting index or reversing traversal does not change the selected endpoint.
This deterministic rule was added after the initial invariance tests exposed
an ordering-dependent nearest-pixel choice.

### 3. Resample and register the arc

`normalize` measures cumulative arc length and interpolates 128 equally spaced
samples. Equal sampling lets short and long contours contribute the same
number of terms to the score.

The first selected contour endpoint becomes the origin. The direction toward
the second endpoint defines the horizontal axis, and its perpendicular defines
the vertical axis. Dividing both coordinates by endpoint distance removes
uniform scale while preserving height relative to base width. In particular,
the code does not independently stretch each arc's height to a common range:
the height difference between extended and folded fingers remains available
to the classifier.

Although this coordinate transform handles translation, orientation and uniform
scale once an arc is selected, the preceding horizontal-corridor selection
assumes the fixed view. The full pipeline has not been validated for arbitrary
hand rotations or unknown endpoints.

### 4. Score both candidate references

One normalized reference is generated for each class. If the observed samples
are `q[i]` and a class reference is `r[c][i]`, its cost is:

```text
cost(c) = (1 / 128) × sum_i ||q[i] - r[c][i]||²
```

This is a dimensionless mean squared Euclidean distance in the normalized
endpoint frame. Lower is better. It is a similarity cost, not a calibrated
probability. The predictor returns the winning label, both costs and the
selected arc. An exact score tie selects `extended` through reference insertion
order.

The predictor receives observed geometry, known endpoints and references.
Ground-truth evaluation labels remain outside its interface.

## Evaluation and results

The reference seeds are 11 for extended and 12 for folded. The recorded default
run uses 100 held-out examples per class: seeds 10000–10099 for extended and
110000–110099 for folded. Reference and evaluation seeds are distinct, but both
splits come from the same synthetic renderer and parameter distributions.

The recorded 200-example run achieved **100% accuracy**, with no
misclassifications. Rows below are true classes; columns are predictions.

| True class | Predicted extended | Predicted folded |
|---|---:|---:|
| Extended | 100 | 0 |
| Folded | 0 | 100 |

Both candidate costs for representative held-out examples are:

| Sample ID | Extended cost | Folded cost | Prediction |
|---|---:|---:|---|
| `heldout_extended_10000` | 0.055582 | 2.705523 | Extended |
| `heldout_folded_110000` | 1.719546 | 0.018369 | Folded |

Both per-class overlays were visually inspected during implementation. The
selected arcs follow the index boundary, with base endpoints at its ends, and
exclude the palm and adjacent fingers. The extended reference is visibly
shorter than the selected held-out extended boundary, illustrating shape
variation despite the correct decision. No failure overlay was generated
because no evaluated example was misclassified.

The overlay uses blue for the selected synthetic boundary, red for the winning
reference mapped back to the supplied endpoint frame, and green for known
endpoints. A limitation of this diagnostic is that the blue line is the arc
returned by the same selection routine used by the predictor, not a separately
annotated boundary. Its legend calls it ground truth, but it does not provide
an independent numerical test of arc-selection accuracy. The visual inspection
and geometric regression assertions supply the present checks on that step.

## Regression checks and saved evidence

The seven passing cases in
[`tests/test_milestone1.py`](../../tests/test_milestone1.py) check:

- Approximately zero self-match cost for each reference.
- Prediction, costs and selected-arc invariance under reversed traversal and
  a shifted contour starting index for both poses.
- Translation and uniform-scale invariance of normalization, with relative
  height preserved.
- A 20-example held-out evaluation, saved CSV rows and readable overlay images.
- Deterministic rendering, binary pixel values and a larger extended silhouette.

These checks cover the controlled workflow; they are not evidence of robustness
to real-image segmentation errors or clutter.

The default output directory is `outputs/milestone1/`. It contains:

| Artifact | Purpose |
|---|---|
| `reference_extended.png`, `reference_folded.png` | Inspect reference silhouettes. |
| `references.npz` | Retain normalized reference arrays. |
| `heldout_*.png` | Retain each evaluated binary silhouette. |
| `scores.csv` | Record IDs, seeds, split, endpoints, truth, prediction and both costs. |
| `summary.json` | Record sample counts, seed ranges, confusion matrix and accuracy. |
| `overlay_extended.png`, `overlay_folded.png` | Inspect representative matches. |
| `overlay_failure.png`, if needed | Inspect the first classification failure. |

Generated outputs are ignored by Git and can be reproduced with the commands
below. Use a fresh output directory when changing the evaluation size, because
the command does not remove images left over from earlier runs.

## Reproducing the experiment on Linux

From the repository root, with Conda installed:

```bash
conda env create -f environment.yml
conda activate hssm
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m hssm.milestone1 \
  --per-class 100 --output outputs/milestone1
```

Alternatively, use Python 3.12 and a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m hssm.milestone1 \
  --per-class 100 --output outputs/milestone1
```

The recorded validation used the virtual-environment workflow. Conda environment
creation was not executed on the cloud machine because Conda was unavailable.

## What milestone 1 establishes

The ordered-contour representation and distance score separate the two
deliberately distinct synthetic classes across the recorded held-out examples.
The code also preserves sample identity, exposes both competing costs and saves
images suitable for inspecting a decision.

The result does not establish general hand recognition performance. Known
endpoints, clean connected silhouettes, fixed viewing geometry and widely
separated height ranges simplify the task substantially. There is no clutter
model, unknown-pose search, multi-finger configuration inference, probabilistic
HSSM decoding or flexion-angle experiment. Those capabilities remain outside
milestone 1.
