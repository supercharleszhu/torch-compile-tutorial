"""
torch.compile 101 — Lesson 3: Graph Breaks (Quiz)
==================================================

A "graph break" happens when torch.compile cannot trace your code into a
single continuous FX graph.  Dynamo compiles what it has so far, runs the
unsupported code eagerly, then starts a NEW graph on the other side.

Why does this matter?
  - Each graph is compiled and optimised independently.
  - Operator fusion, memory planning, and kernel selection all stop at
    graph boundaries — you lose cross-boundary optimisation.
  - More graphs = more compilation overhead at startup.

Common causes of graph breaks:
  1. Data-dependent control flow  — if/while on tensor values
  2. Unsupported Python builtins   — print(), open(), etc.
  3. Tensor → Python escapes       — .item(), .tolist(), .numpy()
  4. Non-traceable third-party code — arbitrary C extensions

QUIZ: The function below has multiple graph breaks baked in.
  - Read the code and predict where each break is.
  - Run the script and check the trace to confirm.
  - Then open 03s_graph_breaks_solution.py to see the fixed version.

Run:  bash run_debug.sh 03
Then: tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite

Reference:
  https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html
"""

import shutil
import torch
from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig

TRACE_DIR = "./torch_trace"
shutil.rmtree(TRACE_DIR, ignore_errors=True)


def noop_pass(gm: torch.fx.GraphModule) -> None:
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
# The buggy function — how many graph breaks can you spot?
# =========================================================================

def buggy_model(x):
    # Step 1: linear transform
    y = x * 2 + 1

    # Step 2: logging (graph break #1 — print is a Python builtin)
    print(f"  intermediate sum = {y.sum()}")

    # Step 3: normalise by mean
    # (graph break #2 — .item() escapes tensor to Python float)
    mean_val = y.mean().item()
    y = y - mean_val

    # Step 4: conditional activation
    # (graph break #3 — data-dependent control flow)
    if y.sum() > 0:
        y = torch.relu(y)
    else:
        y = torch.tanh(y)

    # Step 5: scale
    y = y * 3

    return y


# =========================================================================
# Run it
# =========================================================================

compiled = torch.compile(buggy_model, backend=backend)

print("=" * 60)
print("GRAPH BREAKS QUIZ")
print("How many graph breaks does buggy_model have?")
print("=" * 60)

x = torch.randn(4, 4)
result = compiled(x)
print(f"\noutput shape: {result.shape}")

print("""
ANSWERS:
  Graph break #1 — print()
    print() is a Python builtin Dynamo cannot trace.
    Splits the graph around the print call.

  Graph break #2 — .item()
    Escapes from tensor-land to a Python float.
    The tracer cannot track a Python scalar through
    subsequent tensor ops.

  Graph break #3 — if y.sum() > 0
    The branch condition depends on a tensor VALUE.
    Dynamo cannot know which path to take at trace time,
    so it breaks out to run the condition eagerly.

Total: 3 graph breaks → 4+ separate graphs instead of 1.

See 03s_graph_breaks_solution.py for the fixed version
that compiles into a single graph with zero breaks.
""")

print(f"Trace written to {TRACE_DIR}/")
print("Parse with:  tlparse ./torch_trace/dedicated_log* "
      "-o ./torch_trace_parsed --overwrite")
print("Look for:    multiple graph entries = graph breaks")
