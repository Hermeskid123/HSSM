# Milestone 2: hand structure and edge registration

Milestone 2 replaces the single-finger known-endpoint diagnostic with a
four-finger state graph registered to candidate image edges. It tests two
separate outcomes: where the palm is, and which fingers are extended. The
thumb stays extended. This is a synthetic project adaptation of Wang et al.
(2008), Figure 3 and Sections 4–5, rather than a reproduction of the paper's
learned image model or Section 7 real-image protocol.

## Model and implementation

[`hand_model.py`](../../src/hssm/hand_model.py) defines the graph and maps every
state to a boundary part. The initial state is 1 (right palm), followed by
states 2–4 (thumb). Index, middle and ring fingers each choose an extended
three-state branch or a hidden one-state branch:

| Finger | Extended branch | Hidden branch |
|---|---|---|
| Index | 5 → 6 → 7 | 8 |
| Middle | 9 → 10 → 11 | 12 |
| Ring | 13 → 14 → 15 | 16 |
| Little | 17 → 18 → 20 | 19 → 20 |

State 20 is the only legal end. Figure 3 shares it between the left palm and
the inner side of an extended little finger; the implementation therefore
provides extended/hidden geometry variants with legal predecessor restrictions.
The graph has exactly 16 complete paths. Tests enumerate them independently
and compare them with the 16 configuration traces, including start, end and
every transition. The output `graph.json` records all traces and mappings.

The renderer follows the figure's thumb-right view, keeps the palm and thumb
fixed, and varies each extended finger's height by up to 12 pixels around its
analytic reference. Hidden fingers have shallow boundary arches. It saves
independent boundary/palm annotations before clutter is added. It is a polygonal
silhouette fixture, not a physical hand model. Width and palm shape do not vary
in this experiment, making palm localization substantially easier than on real
hands.

[`milestone2.py`](../../src/hssm/milestone2.py) exposes `infer(mask, condition)`.
It takes neither true endpoints nor true location, orientation, scale or
configuration. OpenCV extracts edges from all components. For each declared
pose hypothesis, canonical state curves supply ordered proposals; proposals
at horizontal offsets −2, 0 and 2 pixels are snapped to nearest observed edge
pixels. Durations of 6 and 8 samples give six candidates per ordinary state;
state 20 has both geometry variants. Every registered point lies on an
extracted image edge, although points can repeat and adjacent registered points
can jump. The restrictions reduce computational cost but also limit the
registrations the decoder can discover.

Shared ordered arc-length resampling was extracted from milestone 1 into
[`contours.py`](../../src/hssm/contours.py). Both milestones now use it. Binary
silhouettes, seeded IDs, OpenCV boundary extraction, headless drawing and
CSV/JSON records retain milestone 1's conventions. A general overlay replaces
the endpoint-specific single-finger overlay for this experiment. Ground-truth
blue curves come directly from the renderer; red curves are the decoded
image-edge chains. The original milestone 1 cohort still produces identical
score CSV contents after this refactor.

## Dynamic programming and chosen score

[`decoder.py`](../../src/hssm/decoder.py) implements a segment-boundary
contraction of the recurrence in Section 5, Eqs. 11–18. Let a segment candidate
`v` for state `j` contain `l` ordered edge features, and let `t` be total features
used so far. The recurrence is:

```text
delta[t,j,l,v] = E(j,l,v) + D(j,l)
                 + min over legal predecessor (i,k,u)
                     {delta[t-l,i,k,u] + A(i,j) + Phi(u,v)}
```

Only state 1 can initialize a path. Only state 20 can terminate it. Backpointers
recover the full sequence of states, selected candidates and durations. Because
the graph is acyclic, topological processing replaces a fixed `Tmax` iteration.
Total registration length remains explicit, so paths with different numbers of
states/features compete at termination.

The paper's internal `beta` minimization (Eq. 13) is restricted here to supplied
ordered candidate chains. Their internal costs are computed before the outer
recurrence. Thus the optimum is exact **over these candidates and the pose
grid**, not over all possible feature permutations or continuous poses.

The implemented additive score is a project design choice:

| Term | Implemented cost |
|---|---|
| Emission | `mean(w * squared_distance(edge, posed_reference)) / 16` |
| State transition `A` | 0 for a legal arc; forbidden otherwise |
| Internal feature transition | `0.01 * mean(squared_error of successive displacement vectors)` |
| Between-segment `Phi` | `0.025 * squared_error of endpoint displacement` |
| Duration `D` | `0.02 * (l - 8)^2 * mean(w)` |
| Initial state | 0 at state 1; all other starts forbidden |
| Unmatched clutter | Constant background term omitted; no learned appearance ratio |

Ordinary extended-finger states have `w=1/3`; hidden branches have `w=1`.
State 20 separately weights its palm and extended little-side samples. Emission
and duration costs are consequently balanced by anatomical region rather than
summed per raw pixel. This differs from the paper's pixel-level likelihood
score. Internal and between-segment penalties still accumulate over states,
so some preference for shorter branches remains. Duration is a sample-count
prior, not the paper's learned, scale-dependent physical duration model.
No SVM, trained emission distribution, or calibrated object-versus-clutter
likelihood is claimed.

The synthetic reference geometry and score constants are fixed, with no fitting
to the held-out cohort. No milestone 3 functionality is included.

## Evaluation protocol

All five conditions contain two held-out examples of each of 16 configurations:
32 images per condition and 160 image evaluations in total. The same shape
seeds are paired across conditions. Configuration bits are ordered index,
middle, ring, little, with `1` meaning extended. Seeds are
`20000 + 100 * configuration_index + repeat`, with lexicographic bit ordering.

The fixed condition uses the canonical pose. Translation varies both offsets
uniformly over ±23 pixels and searches a 49-point grid at 8-pixel spacing.
Rotation varies continuously over ±28 degrees and searches five angles at
15-degree spacing. Scale varies continuously from 0.78 to 1.22 and searches
five scales at 0.125 spacing. Each varies only its named pose factor; the others
are known experimental constants. The clutter condition retains the canonical
pose and adds 28 seeded foreground strokes, including strokes that may merge
with the hand. It is separate from pose variation. These are finite-range,
controlled experiments, not unrestricted image searches.

Palm localization pools 200 samples from each true palm side. A sample is
covered when it lies within **3 pixels** of a densely sampled predicted palm
curve. An image is localized when coverage reaches **75%**. The evaluator uses
true palm annotations; the predictor uses its own state 1/state 20 registration.
The little-finger side is not included in the true palm annotation. Coverage is
directional, so excess predicted boundary is not penalized by this metric.

## Recorded results

| Condition | Palm localized | Mean palm coverage | Exact configuration, all images | Exact on localized images | Mean decode time/image |
|---|---:|---:|---:|---:|---:|
| Fixed | 32/32 (100%) | 100.00% | 23/32 (71.875%) | 71.875% | 0.0186 s |
| Translation | 32/32 (100%) | 99.40% | 25/32 (78.125%) | 78.125% | 0.7779 s |
| Rotation | 32/32 (100%) | 96.02% | 17/32 (53.125%) | 53.125% | 0.0824 s |
| Scale | 32/32 (100%) | 99.04% | 27/32 (84.375%) | 84.375% | 0.0847 s |
| Clutter | 32/32 (100%) | 100.00% | 28/32 (87.500%) | 87.500% | 0.0188 s |

Every image was localized, so the conditional and unconditional configuration
accuracies happen to be equal. The implementation computes their denominators
separately and would report a null conditional accuracy if none localized.
Runtimes include edge extraction, proposal construction, all pose hypotheses
and decoding; they exclude rendering, diagnostic constrained decodes and I/O.
They depend on hardware and timing noise.

| Condition | Index accuracy | Middle accuracy | Ring accuracy | Little accuracy |
|---|---:|---:|---:|---:|
| Fixed | 96.875% | 96.875% | 90.625% | 87.500% |
| Translation | 100% | 96.875% | 96.875% | 84.375% |
| Rotation | 96.875% | 68.750% | 81.250% | 93.750% |
| Scale | 96.875% | 96.875% | 93.750% | 96.875% |
| Clutter | 100% | 96.875% | 96.875% | 93.750% |

Clutter improves exact accuracy on this small paired cohort: five formerly
incorrect seeds become correct and none of the formerly correct seeds become
incorrect. Additional edges alter nearest-edge support and can improve this
particular synthetic score. This result does not establish that clutter helps
recognition generally or that the decoder robustly rejects background edges.

## Inspected failure

`outputs/milestone2/fixed/overlay_failure.png` shows sample `fixed_05_20500`:

- True configuration: `0101` (middle and little extended).
- Predicted configuration: `0100` (little incorrectly hidden).
- Palm coverage: 100%; localization succeeds despite the structure error.
- Best whole-hand cost with little hidden: **1.887**.
- Best whole-hand cost with little extended: **2.776**.

These two costs constrain the little finger at the selected pose and allow the
remaining fingers to optimize. They are complete-registration costs, not local
little-finger probabilities. The overlay records analogous costs for all four
fingers. The hidden branch registers nearby edge samples across the little
finger's base region, while the true finger's height differs from the nominal
reference. Sparse candidate sampling, permitted jumps and the synthetic score
make that wrong branch cheaper. The DP selects the lower cost correctly; this
is a model/candidate limitation, rather than an optimization failure.

Failure overlays from translation, rotation, scale and clutter were also
inspected. The rotation failure demonstrates that palm coverage can remain high
while the decoder misses a visible middle finger. Good palm localization must
therefore not be presented as correct hand-structure recognition.

## Verification and reproduction

All **14 milestone 2 tests** and **21 total tests** pass. Five tiny randomized
problems compare decoded cost and path with independent exhaustive enumeration
of states and segment choices, including negative candidate costs and variable
durations. Another reduced full-hand problem compares against all 16 structures.
Other checks cover the Figure 3 topology, state 20 variants, image-edge
registration without truth inputs, localization thresholds, missing edges and
separation of clutter from pose variation. The final evaluation CSV contains
160 unique IDs, with all 16 configurations represented in every condition.
Both candidate durations occur in the decoded results.

From the repository root on Linux with Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_milestone2.py
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m hssm.milestone2 \
  --per-configuration 2 --output outputs/milestone2
```

With the provided Conda file, run `conda env create -f environment.yml`, activate
`hssm`, and replace `.venv/bin/python` with `python`. Conda creation has not been
executed on this machine because Conda is unavailable. The virtual-environment
workflow above was exercised using the existing Python 3.12 environment.

Results are saved to `outputs/milestone2/summary.json` and `scores.csv`.
Per-condition directories retain observed images, evaluator truth, decoded
points/states/durations, example overlays and failure overlays. `graph.json`
retains topology traces. Generated files are ignored by Git; rerun the command
to regenerate them. Use a fresh directory when changing the cohort size.
