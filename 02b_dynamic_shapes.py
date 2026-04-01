"""
torch.compile 101 — Lesson 2b: Dynamic Shapes
==============================================

Shows how dynamic=True and mark_dynamic avoid recompilation
by using symbolic integers for dimensions.

Run:  bash run_debug.sh 02b
Then: tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
"""

import shutil
import torch
from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig

TRACE_DIR = "./torch_trace"
shutil.rmtree(TRACE_DIR, ignore_errors=True)

# =========================================================================
# 1. dynamic=True: one compilation for all sizes
# =========================================================================
# Dynamo uses symbolic integers for dimensions.  Guards become
# "s0 >= 2" instead of "s0 == 4", so varying sizes reuse the same graph.

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
# =========================================================================
# 2. mark_dynamic: fine-grained control
# =========================================================================
# Mark only the dimensions that actually vary (e.g., batch size).
# The compiler keeps static info on fixed dims for better optimisation.

torch.compiler.reset()

backend2 = DebugBackend(compilation_config=config)


def matmul_fn(x, w):
    return x @ w


compiled_mark = torch.compile(matmul_fn, backend=backend2)

print("--- mark_dynamic on batch dimension only ---")
x1 = torch.randn(4, 16)
torch._dynamo.mark_dynamic(x1, 0)
w = torch.randn(16, 8)
compiled_mark(x1, w)

x2 = torch.randn(32, 16)
compiled_mark(x2, w)

x3 = torch.randn(128, 16)
compiled_mark(x3, w)

print("Batch dim is symbolic — no recompilation.")
print(f"\nTrace written to {TRACE_DIR}/")
print("Parse with:  tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite")
