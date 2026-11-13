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
