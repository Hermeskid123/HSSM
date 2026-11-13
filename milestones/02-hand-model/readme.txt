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
