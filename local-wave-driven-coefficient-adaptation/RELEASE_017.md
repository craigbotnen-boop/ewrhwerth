# Submission Release 017

This snapshot supports the numerical illustration in:

> Craig W. Botnen, *Local Wave-Driven Coefficient Adaptation on Metric Networks: Short-Time Well-Posedness and Finite Propagation*.

## Release 017 changes

- the external endpoint condition is the analytical reflecting condition `p=0`, implemented as `r=-ell` at every SSP-RK stage;
- the solver records the maximum pre-clipping coefficient correction;
- the coefficient projection was inactive on all five grids (`maximum_projection_correction_all_grids = 0.0`);
- the detection threshold remains `1e-10`;
- the numerical illustration remains separate from the analytical proof.

Craig W. Botnen is the sole author and is responsible for the mathematics, code, citations, interpretation, and submission decisions. AI-assisted tools were used for language, typesetting, reproducibility support, and structured internal critique. This repository does not claim external human peer review.
