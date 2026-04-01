"""
torch.compile 101 — Lesson 3 Solution: Zero Graph Breaks
=========================================================

This is the fixed version of buggy_model from 03_graph_breaks.py.
All three graph breaks have been eliminated:

  1. print()          → removed (or use @torch.compiler.disable)
  2. .item()          → keep computation as tensor ops
  3. if y.sum() > 0   → torch.cond (traces both branches)

Run:  bash run_debug.sh 03s
Then: tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite

Compare the trace to 03_graph_breaks.py — this version should
produce a single graph entry instead of 4+.
"""

import shutil
import torch
from linductor.compiler_backend import (
    LinductorBackend,
    LinductorCompilationConfig,
    PassConfig,
)

TRACE_DIR = "./torch_trace"
shutil.rmtree(TRACE_DIR, ignore_errors=True)


def noop_pass(gm: torch.fx.GraphModule) -> None:
    gm.graph.lint()
    gm.recompile()


linductor_config = LinductorCompilationConfig(
    inductor_config={},
    pass_config=PassConfig(graph_pass=noop_pass),
    torch_trace_enabled=True,
    torch_trace_dir=TRACE_DIR,
)
linductor_backend = LinductorBackend(compilation_config=linductor_config)


# =========================================================================
# The fixed function — single graph, zero breaks
# =========================================================================

def fixed_model(x):
    # Step 1: linear transform (same as before)
    y = x * 2 + 1

    # Step 2: logging removed
    # Fix #1: print() was causing a graph break.
    # If you need logging, move it outside the compiled function
    # or use @torch.compiler.disable on a separate helper.

    # Step 3: normalise by mean
    # Fix #2: keep mean as a tensor instead of calling .item()
    y = y - y.mean()

    # Step 4: conditional activation
    # Fix #3: use torch.cond to trace both branches symbolically
    y = torch.cond(
        y.sum() > 0,
        lambda y: torch.relu(y),
        lambda y: torch.tanh(y),
        (y,),
    )

    # Step 5: scale (same as before)
    y = y * 3

    return y


# =========================================================================
# Run it
# =========================================================================

compiled = torch.compile(fixed_model, backend=linductor_backend)

print("=" * 60)
print("GRAPH BREAKS SOLUTION — zero breaks")
print("=" * 60)

x = torch.randn(4, 4)
result = compiled(x)
print(f"\noutput shape: {result.shape}")

print("""
Fixes applied:
  1. print()        → removed from compiled function
  2. .item()        → y - y.mean()  (stays as tensor op)
  3. if/else        → torch.cond()  (traces both branches)

Result: single graph, no breaks.
""")

print(f"Trace written to {TRACE_DIR}/")
print("Parse with:  tlparse ./torch_trace/dedicated_log* "
      "-o ./torch_trace_parsed --overwrite")
print("Compare:     single graph entry vs 4+ in the buggy version")
