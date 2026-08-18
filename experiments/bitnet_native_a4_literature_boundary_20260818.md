# Native BitNet A4 literature boundary — 2026-08-18

## Binding experimental status

- Gate009 closed the separately coded radius / shape-gain compression-method lane. Do not reopen it.
- Gate010 is the active transfer test at the true `BitLinear.activation_quant` sites of `microsoft/bitnet-b1.58-2B-4T`.
- Gate011 is preregistered and may run only if Gate010 passes. It tests the BitNet-v2-style per-token absmean INT4 scalar rule at the same frozen BitLinear sites.
- Gate012 is preregistered and may run only if Gate011 passes. It tests a direct Four-Over-Six-inspired two-scale MSE selector at the same sites.
- Gate012 is terminal for the adaptive-codebook method claim. Do not add further method-saving gates after a Gate012 failure.

## Literature boundary

### BitNet v2 — arXiv:2504.18415

BitNet v2 uses per-token absmax for INT8 activations and per-token absmean for INT4 activations. Its INT4 rule is

`beta = mean(abs(X)); Q_INT4(X) = beta/sqrt(7) * RoundClip(sqrt(7)/(beta+eps) * X, -8, 7)`.

BitNet v2 also introduces H-BitLinear on selected projections and continue-trains the architecture with INT4 activations. Therefore Gate011 is only a scalar-rule hostile control on the frozen b1.58 checkpoint, not a comparison to the full BitNet-v2 architecture.

### Four Over Six — arXiv:2512.02010

Four Over Six evaluates two candidate block-local scales, quantizes under both, and selects the representation by reconstruction error. The paper reports MSE as the best overall scale-selection rule and explicitly studies activation tensors as well as weights and gradients; its PTQ tables include W4A4 experiments. Therefore adaptive multi-scale MSE selection is prior art at the mechanism level.

Gate012 is deliberately labeled `4/6-inspired INT4`, not an implementation of NVFP4 Four Over Six. It freezes two token-local INT4 clip candidates with a 4:6 range ratio and selects by full-vector MSE.

### BlockDialect — ICML 2025

BlockDialect assigns blocks to representations from a small formatbook and includes an online activation-quantization procedure. Small discrete representation sets plus online selection are therefore not novel in the broad sense.

### AQuant — arXiv:2208.11945

AQuant establishes runtime-adaptive activation quantization via activation-dependent rounding decisions. It is not the same mechanism as a scale dictionary, but it further pre-empts broad claims around online adaptive activation quantization.

### AAAC — arXiv:2605.08692

AAAC uses small adaptive codebooks and reconstruction-oriented selection for 4-bit LLM weight quantization. It applies to weights rather than per-token activations, but it makes generic codebook-selection novelty claims unsafe.

### AdaMX — arXiv:2608.03867

AdaMX adapts low-bit representation / precision-recovery choices to block heterogeneity, including operand-specific activation encoding. Broad heterogeneity-aware low-bit adaptation is occupied.

## Surviving possible claim, only if Gates010–012 all pass

A defensible contribution would need to be narrowly stated as:

> On a frozen native ternary BitNet checkpoint, calibration-derived per-BitLinear INT4 scale dictionaries improve activation quantization at the true BitLinear sites beyond both the BitNet-v2-style absmean scalar rule and a purely token-local multi-scale MSE selector.

This would still require larger/task-level evaluation and rate/metadata accounting. It would not establish broad novelty for adaptive clipping, MSE scale selection, codebooks, W1.58A4, or activation heterogeneity.

## Terminal interpretation

- Gate010 fail: stop the adaptive-A4 method lane; retain prior results as representation diagnostics.
- Gate010 pass + Gate011 fail: baseline choice explains the effect; stop the method claim.
- Gates010–011 pass + Gate012 fail: known adaptive scale-selection principles explain the effect; stop the method claim.
- Gates010–012 all pass: proceed to larger/task-level evaluation under the narrow claim above.
