"""End-to-end Inductor pipeline walkthrough.

Runs the Lesson 4 model under torch.compile with `inductor_hooks` installed
so every interesting stage prints a one-line trace. Companion to:

    https://supercharleszhu.github.io/posts/2026-04-30-00/

Usage:
    cd /home/chzhu/personal/torch-compile-tutorial/deep_dive_ii_inductor_internals
    python run_walkthrough.py 2>&1 | grep '\\[hooks\\]'

Filter further by stage with:
    python run_walkthrough.py 2>&1 | grep 'Stage 2:'
"""
import os
import sys
import shutil

# Make sibling debug_backend.py importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig
import inductor_hooks

torch.compiler.reset()

CACHE_DIR = os.path.abspath("./_inductor_cache_walkthrough")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
shutil.rmtree(CACHE_DIR, ignore_errors=True)

# Install all hooks. Set pause=True if you want to drop into pdb at each stop.
inductor_hooks.install(pause=False)


def fn(a, b, c):
    """Lesson 4's tiny example: three pointwise + one reduction."""
    return ((a + b) * c.relu()).sum(dim=-1)


backend = DebugBackend(
    compilation_config=DebugCompilationConfig(
        inductor_config={},
        pass_config=PassConfig(),
        torch_trace_dir=os.path.abspath("./torch_trace"),
        torch_trace_enabled=False,   # the hooks already print everything
        write_visualized_graph=False,
    )
)

device = "cpu"
a = torch.randn(64, 128, device=device, dtype=torch.float32)
b = torch.randn(64, 128, device=device, dtype=torch.float32)
c = torch.randn(64, 128, device=device, dtype=torch.float32)

print("=" * 78)
print("Compiling fn = ((a + b) * c.relu()).sum(-1)")
print("=" * 78)

compiled = torch.compile(fn, backend=backend, fullgraph=True)
out = compiled(a, b, c)
torch.testing.assert_close(out, fn(a, b, c), rtol=1e-3, atol=1e-3)

print("\n" + "=" * 78)
print("Done. Map the [hooks] lines above to blog stages 1-6.")
print(f"Generated wrapper: {CACHE_DIR}/<hash>/output_code.py")
print("=" * 78)
