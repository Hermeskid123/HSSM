Milestone 3 | At what flexion angle does the preferred path switch?
===================================================================

Use an articulated renderer or a dataset with known joint angles. Select one
finger and joint, document the flexion axis and angle convention, and render
a sweep from extended to folded. Hold the camera, shape, other joint angles,
lighting and background fixed within a sweep. Record degrees, projected
contours and the render settings for every frame.

Freeze the matcher from milestone 2. For each frame, decode the best path
with the chosen finger constrained to extended, then to hidden. Keep other
settings identical. Express both scores as costs where lower is better. Save
cost_hidden minus cost_extended. A positive value favors extended. Report
ties, every sign change, and sweeps with no crossing. Inspect neighboring
frames when a switch occurs.

Output a per-frame CSV, a score-margin plot against angle, and overlays before
and after each switch. Intermediate poses can be genuinely ambiguous. Treat
a switch as a model decision and report binary accuracy only for frames with
an independently defined visibility label. Repeat across hand shapes to test
whether the transition angle generalizes.
