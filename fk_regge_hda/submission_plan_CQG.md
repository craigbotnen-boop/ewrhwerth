# Submission plan — Classical and Quantum Gravity first

## Recommended target

**First submission:** Classical and Quantum Gravity (CQG), Research Paper.

Reason: CQG explicitly lists classical GR, canonical formalisms, Regge calculus, dynamical triangulation and related simulation methods within scope. The manuscript is narrowly centered on canonical/refinement structure in Regge gravity and cites a substantial CQG lineage.

**Fallback:** Physical Review D, Research Article. PRD also covers general relativity and quantum gravity and has no length limit for regular research articles, but the current paper is more naturally embedded in the specialist Regge/canonical-gravity audience served by CQG.

## Submission positioning

Do not sell the paper as discovery of quadratic gauge breaking. State that prior Regge work established:

- exact/nonlinear and contracted Bianchi identities;
- small-deficit approximate conservation laws;
- quadratic lifting of would-be gauge Hessian modes with deficit angle;
- exact flat/homogeneously-curved discrete HDA sectors.

The manuscript contribution is the **stationary length-Regge refinement realization through the full canonical reduction**:

- H_gg = O(h^4) on a fully stationary FK48 family;
- K_gg = O(h^4) directly in the old/new mixed canonical block;
- K_gp = O(h^2), establishing sector-specific rather than blanket restoration;
- bounded stationary response on geometric deformation sources, with h^-2 hostile control for unrestricted physical forcing;
- direct inverse-metric FK structure function;
- fixed-volume weak assembly.

## Required CQG/IOP policy items

### Research-data statement

Before final submission, the public reproducibility archive should include the source needed to regenerate the stationary FK48 branch, Hessians, mixed boundary Jacobians, raw numerical tables and figures. Prefer a DOI-bearing archive (e.g. Zenodo snapshot of the GitHub release) in addition to the live repository.

Draft statement once the DOI exists:

> **Data availability.** All numerical data and source code required to reproduce the stationary FK48 calculations, Hessian and mixed-block spectra, refinement fits, and figures reported in this article are openly available in the accompanying reproducibility archive at [DOI]. The development repository is available at [repository URL].

Do not use this final wording until the archive actually contains the full stationary solver.

### Generative-AI disclosure

IOP currently requires disclosure when generative-AI tools are used for text generation/editing, literature support, or generating figures from existing data.

Proposed acknowledgement wording, to be reviewed by the author before submission:

> **AI-assisted research and drafting.** OpenAI ChatGPT (GPT-5.6 Sol) was used as an interactive computational and writing assistant for symbolic algebra checks, numerical-analysis scripting, literature-search support, manuscript organization and language drafting, and generation of plots from author-controlled numerical data. All mathematical claims, calculations, references, numerical results and manuscript text were reviewed and accepted by the author, who takes full responsibility for the work.

Because IOP's policy states that AI must not fabricate or manipulate original research results, the reproducibility archive should make clear which numerical results arise from explicit Regge computations and which plots are presentation layers generated from frozen numerical data.

## Final pre-submission gates

1. Independent human proof review of theorem_v2.md.
2. Complete clean-room stationary FK48 solver and regenerate headline numbers from a fresh checkout.
3. Add DOI-bearing reproducibility archive and cite it.
4. Insert data-availability and AI-disclosure sections into the journal manuscript.
5. Convert to the current CQG/IOP submission template only after content freeze.
6. Cross-check every reference and equation anchor against the cited primary paper.
7. Verify all plots from raw machine-readable files.
8. Final terminology sweep: length Regge; on shell; Euclidean canonical theorem; Lorentzian kinematics only.

## PRD fallback notes

PRD regular Research Articles currently have no length limit. APS requires a Data Availability Statement and a disclosure of substantive AI use. A PRD version should use REVTeX and emphasize the broader implication for restoration of canonical covariance rather than the specialist Regge-calculus lineage.
