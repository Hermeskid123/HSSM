# Project instructions

Build a CPU-runnable Python implementation for studying variable hand contours.
Read the selected milestone's readme.txt before editing. Begin with milestone 1
when a request says to start the project. Each milestone has its own scientific
question and evaluation.

## Source and scope

Use references/wang_pami2008.pdf, Wang, Athitsos, Sclaroff and Betke (2008),
as the primary reference. Figure 3 shows the hand topology. Sections 4–6
describe the probabilistic formulation, registration and implementation.
Section 7 describes the experimental protocol. Cite the relevant section or
equation when implementing a paper component. Label design choices such as
known endpoints, synthetic fixtures and simplified matching as project
adaptations. Reserve claims of reproduction for matched methods and protocols.

## Implementation conventions

Use NumPy and OpenCV for the first diagnostic. Keep rendering, inference and
evaluation callable independently. Use (x, y) pixel coordinates, (row, column)
array indexing and explicit angle units. Carry sample IDs and seeds from
generation through scoring and visualization. Keep evaluation labels outside
the predictor interface. The known endpoints in milestone 1 are deliberate
inputs. Run inference on CPU and write image outputs without a display.

Keep reusable code under src/, regression checks under tests/, and generated
runs under outputs/. Each implemented milestone should expose a command that
works from the repository root. Add actual install, run and test commands to
its readme.txt as part of implementation.

## Verification and handoff

Run the documented commands. Report the evaluated split, number of examples,
metrics, output paths and any observed failure cases. Use blue for ground truth
and red for predicted matches and add a legend. Include scores for both
candidate paths when diagnosing a finger decision. Distinguish a test on
synthetic contours from a result on the paper's real-image dataset.
