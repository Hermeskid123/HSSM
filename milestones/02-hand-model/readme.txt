Milestone 2 | Which fingers are visible, and where is the hand?
===============================================================

Use Figure 3 to encode the palm, thumb and four finger decisions in an
explicit transition graph. Keep the thumb extended. Trace all 16 assignments
through the graph and check the start, end and legal transitions against the
figure. Record the state-to-boundary mapping in the implementation.

Replace the known-endpoint matcher with registration to candidate image
edges. Implement the paper's dynamic programming recurrence and backtracking
for the chosen score. Identify the emission, transition, duration and clutter
terms included in each run. Document any approximation, particularly any
change to scores for paths of different lengths. Verify the decoded score and
path against exhaustive enumeration on a tiny graph with few candidate edges.

Begin with known orientation and scale, then vary translation, rotation,
scale and added clutter in separate runs. Evaluate held-out shapes across all
16 configurations. Keep true location, scale, orientation and structure in
the evaluator for the corresponding unknown-parameter runs.

Report palm localization and exact configuration separately. For synthetic
data, define a pixel tolerance and measure the fraction of true palm-edge
samples within it of a predicted palm edge. Count localization as successful
at 75 percent coverage and record the tolerance. Report per-finger accuracy
and exact configuration accuracy across all examples, plus exact accuracy on
localized examples. Show a failure overlay and record runtime per image.

Implemented milestone 2 workflow
--------------------------------
From the repository root on Linux, using Python 3.12:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_milestone2.py
    PYTHONPATH=src .venv/bin/python -m pytest -q
    PYTHONPATH=src .venv/bin/python -m hssm.milestone2 --per-configuration 2 --output outputs/milestone2

Conda alternative:
    conda env create -f environment.yml
    conda activate hssm
    PYTHONPATH=src python -m pytest -q
    PYTHONPATH=src python -m hssm.milestone2 --per-configuration 2 --output outputs/milestone2

No additional dependencies, display, GPU, service or credentials are required.
The Conda creation command remains unverified here (Conda is unavailable).
Use a fresh output directory when changing cohort size to avoid stale images.
The module runs all five conditions and writes graph.json (mapping, transitions,
and all 16 legal traces), scores.csv, summary.json, observed PNGs, per-image JSON
with independent renderer truth and decoded state/edge chains, and example and
failure overlays in condition subdirectories. Bits are index, middle, ring,
little; 1=extended, 0=hidden. The thumb is always extended. Seeds and sample IDs
are retained. Timing covers edge extraction, all pose hypotheses and decoding,
but excludes rendering, diagnostic constrained reruns and output writing.

Code organization and reuse
---------------------------
src/hssm/hand_model.py: Figure 3 graph, explicit state-to-boundary mapping,
    configuration traces, independently sampled hand geometry and renderer.
src/hssm/decoder.py: generic segmental DP and backtracking over supplied
    segment candidates, retaining total registration length t and duration.
src/hssm/milestone2.py: image-edge proposals, pose search, inference interface,
    localization metric, evaluation and command line.
src/hssm/contours.py: arc-length resampling extracted from milestone 1 and used
    by both milestones, plus general blue/red headless overlays.
tests/test_milestone2.py: exhaustive decoder checks, topology checks and image
    registration/metric checks.

The milestone 1 single-finger renderer and endpoint-specific overlay cannot
represent all four fingers or unknown pose. Milestone 2 therefore adds a full
hand renderer and general overlay while retaining binary-mask data, OpenCV
contour extraction, seeded generation, coordinate conventions, CSV/JSON output
and the shared resampling routine. The original milestone 1 evaluator still
runs with byte-identical scores on its recorded 200-example cohort.

Paper correspondence and restrictions
-------------------------------------
See report.md for equations, assumptions, full results and a failure diagnosis.
The graph follows Wang et al. (2008), Figure 3, with thumb-hidden shortcuts
excluded. State 20 is the sole end state and includes the little-finger inner
side when extended, as well as the left palm boundary. Its two candidate
realizations are restricted to predecessors 18 (extended) and 19 (hidden).

The decoder implements the segment-boundary form of Section 5, Eqs. 11-18,
for an additive project score. It is exact over the supplied finite candidate
chains and pose grid, not over arbitrary image-feature orderings. Each segment
has 6 or 8 ordered image-edge samples, proposed by snapping a posed canonical
curve with x offsets -2, 0, 2 pixels to nearest extracted edges. No known base
endpoints or true pose/configuration enter infer(mask, condition). All image
components are eligible; repeated edge-pixel assignments are permitted.

Costs are synthetic and not fitted to held-out data: spatial residual emission,
legal state transitions (0 or infinity), inter/intra-segment displacement
penalties and a preferred sample-count duration of 8. There is no learned
object/clutter likelihood ratio or SVM; unused edges contribute a constant
omitted background cost. Segment means and branch weights change the paper's
pixel-sum score to reduce path-length bias. Some bias remains because internal
and inter-segment penalties are summed over selected states. Duration counts
are quadrature samples, not a learned physical edge-length distribution. This
is a restricted synthetic adaptation, not a reproduction of Sections 4-7.

Five isolated evaluation conditions
----------------------------------
Each has 32 held-out images: 2 examples of each of the 16 configurations.
Analytic reference geometry is fixed; heights vary independently by +/-12
pixels in held-out generation. Seed=20000 + 100*configuration_index + repeat;
configuration_index follows lexicographic bit order. Shapes are paired across
conditions. Pose parameters remain in evaluator records; the predictor receives
only the image and the experimental search condition. The fixed conditions
assume the declared canonical pose, not an image-dependent truth estimate.

fixed: canonical location, orientation and scale, no clutter; one pose.
translation: true dx,dy uniform in [-23,23] pixels; grid -24..24 in steps of 8;
    orientation 0 degrees and scale 1 are known (49 hypotheses).
rotation: true angle uniform in [-28,28] degrees; candidate angles
    -30,-15,0,15,30 degrees; canonical location and scale known (5 hypotheses).
scale: true uniform scale in [0.78,1.22]; candidate scales
    0.75,0.875,1,1.125,1.25; canonical location/orientation known (5 hypotheses).
clutter: canonical pose, 28 seeded foreground line strokes that may overlap
    the hand; no pose variation (one hypothesis). No separate erasure/occluder
    or real-background dataset is modeled.

Palm localization: pool 200 uniformly spaced samples from each of the two true
palm sides; measure the fraction within 3 pixels of a registered predicted palm
curve (densely resampled). Success requires at least 75% coverage. Exclude the
little-finger side from ground-truth palm annotations. Prediction derives its
palm portion from the selected state-20 candidate, not the true structure.
Report exact configuration over ALL images and separately over localized
images; never discard localization failures before computing overall accuracy.

Recorded validation
-------------------
14 milestone 2 tests and 21 total tests passed. Five randomized tiny candidate
problems matched exhaustive enumeration in both optimum score and backtracked
path; a reduced full-hand problem also matched enumeration of all 16 structures.
The default evaluation completed 160 images, with all 16 configurations present
in every condition. Palm localization passed on 32/32 in each condition; exact
configuration counts were fixed 23/32, translation 25/32, rotation 17/32,
scale 27/32 and clutter 28/32. See report.md for per-finger rates, runtimes,
coverage and the observed little-finger failure. These figures concern clean
synthetic geometry and the specific finite search grids only. Stop here;
no milestone 3 flexion experiment has been implemented.
