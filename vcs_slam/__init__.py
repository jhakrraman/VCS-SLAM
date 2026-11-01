"""VCS-SLAM: Visibility-Consistent Semantic Fusion for 3D Gaussian SLAM.

The SLAM backbone (tracking, mapping, densification, keyframes) comes from SplaTAM and
SGS-SLAM. This package holds what VCS-SLAM adds on top of it:

* ``reliability`` - the geometry-validated semantic objective (VCSU, SCEA, CAUW; Eqs. 3-13)
* ``semantics``   - the per-Gaussian semantic logit field and its label space (Sec. III-A)
* ``metrics``     - semantic evaluation, incl. the mechanism metrics of Tab. VII
* ``priors``      - structured corruption of the 2D semantic priors (Tab. VIII)
* ``config``      - experiment config loading with command-line overrides
"""
