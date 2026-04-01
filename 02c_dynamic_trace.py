"""
torch.compile 101 — Lesson 2c: Tracing Dynamic Shapes with Linductor
=====================================================================

Captures a full Linductor trace with dynamic shapes enabled so you can
see how symbolic dimensions propagate through the compilation pipeline.

Run:  bash run_debug.sh 02c
Then: tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
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

torch.compiler.reset()

def noop_pass(gm: torch.fx.GraphModule) -> None:
    """No-op pass — validates the graph without modifying it."""
    gm.graph.lint()
    gm.recompile()


linductor_config = LinductorCompilationConfig(
    inductor_config={},
    pass_config=PassConfig(graph_pass=noop_pass),
    torch_trace_enabled=True,
    torch_trace_dir=TRACE_DIR,
)
linductor_backend = LinductorBackend(compilation_config=linductor_config)


def compute(x):
    return (x * 2 + 1).sum()


compiled = torch.compile(compute, backend=linductor_backend, dynamic=True)

print("--- Linductor trace with dynamic shapes ---")
compiled(torch.randn(10))
compiled(torch.randn(50))
compiled(torch.randn(200))
print(f"\nTrace written to {TRACE_DIR}/")
print("Parse with:  tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite")
