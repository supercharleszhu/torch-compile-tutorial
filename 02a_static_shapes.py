"""
torch.compile 101 — Lesson 2a: Static Shapes and Recompilation
===============================================================

Demonstrates how Dynamo specialises on exact tensor shapes by default,
causing a full recompilation every time a new shape is seen.

Run:  bash run_debug.sh 02a
Then: tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
"""

import shutil
import torch
from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig

TRACE_DIR = "./torch_trace"
shutil.rmtree(TRACE_DIR, ignore_errors=True)

torch.compiler.reset()

def noop_pass(gm: torch.fx.GraphModule) -> None:
    """No-op pass — validates the graph without modifying it."""
    gm.graph.lint()
    gm.recompile()


config = DebugCompilationConfig(
    inductor_config={},
    pass_config=PassConfig(graph_pass=noop_pass),
    torch_trace_enabled=True,
    torch_trace_dir=TRACE_DIR,
)
backend = DebugBackend(compilation_config=config)


def fn(x):
    return (x * 2).sum()


# Static shapes (default): Dynamo specialises on the exact shape seen during
# tracing.  Each new shape fails the guard and triggers a full recompile.
compiled = torch.compile(fn, backend=backend)

print("--- Static shapes (default) ---")
compiled(torch.randn(4))      # compile #1: shape guard records (4,)
compiled(torch.randn(4))      # guard passes → cached version reused
compiled(torch.randn(8))      # shape (8,) ≠ (4,) → recompile #2
compiled(torch.randn(8))      # guard passes → reused
compiled(torch.randn(16))     # shape (16,) ≠ (4,) or (8,) → no recompile because automatic dynamic shape

print("\nEach unique shape triggers a full re-trace + re-compile.")
print(f"Trace written to {TRACE_DIR}/")
print("Parse with:  tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite")
