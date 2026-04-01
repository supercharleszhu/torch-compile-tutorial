"""
torch.compile 101 — Lesson 1: Compile Basics
=============================================

Topics covered:
  - What torch.compile does and why it matters
  - Using the Linductor backend with tracing enabled
  - First-call compilation vs. cached execution
"""

import shutil
import torch
import torch.nn.functional as F
from linductor.compiler_backend import (
    LinductorBackend,
    LinductorCompilationConfig,
    PassConfig,
)

torch.compiler.reset()

# ---------------------------------------------------------------------------
# Linductor backend
# ---------------------------------------------------------------------------
# Linductor is LinkedIn's custom compilation backend that wraps TorchInductor.
# It provides:
#   1. Structured trace export — redirects TORCH_TRACE logs to a directory so
#      every compilation stage (Dynamo bytecode analysis, AOTAutograd graph
#      capture, Inductor IR lowering, kernel codegen) is recorded in one place.
#   2. Custom graph pass hooks — a PassConfig lets you inject graph-level
#      transformations (in-place passes or fx.Transformer subclasses) that run
#      before Inductor's own optimisation pipeline.  When tracing is enabled,
#      DOT/SVG files are generated showing before/after state of each pass.
#   3. Inductor config validation — the config dict is checked against
#      torch._inductor's known options at init time, catching typos early.
#
# Under the hood, LinductorBackend.__call__ applies your custom passes, then
# delegates to torch._inductor.compile_fx.compile_fx with the validated
# config_patches — so the actual code generation (Triton/C++/OpenMP) is still
# handled by upstream Inductor.
#
# Traces are written to TRACE_DIR below.  After running, parse them with:
#   tlparse ./torch_trace -o ./torch_trace_parsed
# ---------------------------------------------------------------------------
TRACE_DIR = "./torch_trace"

# Clean up previous trace logs so each run starts fresh.
shutil.rmtree(TRACE_DIR, ignore_errors=True)


def noop_pass(gm: torch.fx.GraphModule) -> None:
    """A no-op graph pass that simply validates and recompiles the graph.

    This serves as a skeleton for writing custom passes.  When tracing is
    enabled, Linductor's GraphTransformObserver will still generate
    before/after DOT files — useful for verifying the pass infrastructure
    works without actually modifying the graph.
    """
    gm.graph.lint()
    gm.recompile()


linductor_config = LinductorCompilationConfig(
    inductor_config={"max_autotune": True},
    pass_config=PassConfig(graph_pass=noop_pass),
    torch_trace_enabled=True,
    torch_trace_dir=TRACE_DIR,
    write_visualized_graph=True,
)
linductor_backend = LinductorBackend(compilation_config=linductor_config)


def compute(x, y):
    """A function with several ops that Dynamo will trace into an FX graph."""
    a = x + y
    b = a * 2.0
    c = torch.relu(b)
    e = F.silu(c)       # decomposed by AOTAutograd into sigmoid + mul
    d = e.sum(dim=1)
    return d.mean()


# Compile and run
x = torch.randn(100, 100, requires_grad=True)
y = torch.randn(100, 100)

compiled = torch.compile(compute, backend=linductor_backend)

result = compiled(x, y)
print(f"Result: {result.item():.4f}")
