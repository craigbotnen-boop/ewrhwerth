# Gate 010 quantizer audit — 2026-08-18

Status: PASS_AUDIT_NO_PROTOCOL_CHANGE

This audit was performed after Gate 010 launch and does not change the frozen experiment.

## Native BitNet / Transformers scalar rule

Current `transformers.integrations.bitnet.BitLinear.activation_quant` defines signed integer bounds

- `Qn = -(2 ** (num_bits - 1))`
- `Qp = 2 ** (num_bits - 1) - 1`
- `scale = Qp / absmax(token)`

so the direct 4-bit absmax analogue has `Qn=-8`, `Qp=7`, and `scale=7/absmax`. Gate 010 uses exactly that convention for `native_site_dynamic_absmax_A4`.

Therefore the Gate 010 absmax-A4 baseline does **not** require correction from 7/absmax to 8/absmax.

## Gate 011 scalar hostile control

BitNet v2 uses a different INT4 rule: per-token absmean rather than absmax. The preregistered Gate 011 rule remains:

`beta = mean(abs(x))`

`scale = sqrt(7)/(beta + eps)`

`q = round(x*scale).clip(-8,7)`

with dequantization through the existing BitLinear post-quantization path. This is a scalar-rule hostile control only; it is not a reproduction of the full BitNet-v2 H-BitLinear + continue-training architecture.

## Binding decisions

1. Gate 010 remains unchanged and binding.
2. Gate 011 remains on hold unless Gate 010 passes.
3. Gate 012 remains on hold unless Gate 011 passes.
4. Gate 009 remains binding: do not reopen the separate shape-gain/radius codec lane.
5. No additional rescue gates are to be added after Gate 012.
