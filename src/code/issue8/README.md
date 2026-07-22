# MoE Combine Three-Mode Precision/Traffic Trade-off Analysis

Analysis and decision framework for DeepEP V2's three combine reduction modes:
mode A (no-expand), mode B (expand + local reduction), and mode C (expanded send).

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib scipy

# Run the analytical model
python combine_mode_analysis.py --experts 64 --topk 8 --hidden 7168 --repetition-rate 0.3

# Generate decision table
python combine_mode_analysis.py --generate-decision-table

# Run hardware benchmark (requires DeepEP + multi-GPU)
python test_compare_modes.py --num-experts 64 --num-tokens 4096
```

## Files

| File | Purpose |
|------|---------|
| `combine_mode_analysis.py` | Analytical model: traffic, latency, precision error estimation |
| `decision_model.py` | Decision table generator for mode selection |
| `test_compare_modes.py` | Hardware benchmark script (requires DeepEP environment) |

## Three Combine Modes

### Mode A: No-Expand (no local reduce)
- One return token maps to one source, direct load/store write-back
- **Traffic**: O(num_tokens × hidden), each token from exactly one rank
- **Precision**: Best — no intermediate accumulation, no cancellation
- **SM usage**: Minimal — single TMA load + store per token

### Mode B: Expand + Allow Multiple Reduction
- Local shared-memory vectorized reduction of multiple top-k copies
- **Traffic**: O(num_tokens × hidden) reduced by factor of (1/repetition_rate)
- **Precision**: Good — BF16 accumulation in shared memory, some rounding
- **SM usage**: Moderate — reduction loops with hadd bypass when topk ≤ 2

### Mode C: Expanded Send (no local reduce)
- Each copy sent separately, epilogue does final reduction
- **Traffic**: O(num_tokens × topk × hidden) — highest
- **Precision**: Excellent — FP32 accumulation in epilogue
- **SM usage**: Highest — multiple TMA operations, no hadd bypass

## Decision Logic (simplified)

```
if repetition_rate < 0.1:
    → Mode A (no-expand) — minimal overhead, best precision
elif precision_requirement == "strict" and topk_repetition > 0.5:
    → Mode C (expanded send) — FP32 accuracy
else:
    → Mode B (expand + reduce) — balanced traffic/precision
```
