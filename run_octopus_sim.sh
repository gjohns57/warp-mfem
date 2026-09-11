uv run -m mfem.refinement.sim \
    --iterations 12 \
    --substeps 1 \
    --mu 5.0 \
    --fps 24 \
    --contact-d0 1.0e-3 \
    --contact-d1 3.0e-3 \
    --contact-stiffness 1e3 \
    -gplr
