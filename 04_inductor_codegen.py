"""
torch.compile Deep Dive — Lesson 4: Inductor Codegen Walkthrough
================================================================

Topics covered:
  - How a tiny FX graph becomes a single fused kernel
  - Where bufN allocations, kernel calls, and `del` lines come from
  - Reading output_code.py side-by-side with 04_inductor_codegen.md

Companion read: 04_inductor_codegen.md (this directory) and
the blog post "Torch.compile Deep Dive II — Inductor Codegen and
Buffer Lifetimes, End to End".

Run:
    python 04_inductor_codegen.py
    # parse the trace:
    tlparse ./torch_trace/dedicated_log_torch_trace_*.log \\
        --overwrite -o ./torch_trace_parsed
    # then open ./torch_trace_parsed/index.html
"""

import os
import shutil
import torch
from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig

torch.compiler.reset()

TRACE_DIR = os.path.abspath("./torch_trace")
shutil.rmtree(TRACE_DIR, ignore_errors=True)
os.makedirs(TRACE_DIR, exist_ok=True)

CACHE_DIR = os.path.abspath("./_inductor_cache_lesson04")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = CACHE_DIR
shutil.rmtree(CACHE_DIR, ignore_errors=True)


def fn(a, b, c):
    """Three pointwise ops + one reduction. Should fuse into ONE kernel.

    Expected output_code.py shape:
        def call(args):
            arg0_1, arg1_1, arg2_1 = args
            args.clear()
            buf0 = empty_strided_cpu((64,), (1,), torch.float32)
            cpp_fused_add_mul_relu_sum_0(arg0_1, arg1_1, arg2_1, buf0)
            del arg0_1
            del arg1_1
            del arg2_1
            return (buf0,)
    """
    return ((a + b) * c.relu()).sum(dim=-1)


backend = DebugBackend(
    compilation_config=DebugCompilationConfig(
        inductor_config={},  # vanilla — picked tile stays deterministic
        pass_config=PassConfig(),
        torch_trace_dir=TRACE_DIR,
        torch_trace_enabled=True,
        write_visualized_graph=True,
    )
)

# CPU is fine for this lesson — the codegen story is the same as CUDA modulo
# `in_ptr0 + tl.load(...)` vs. `Vectorized<float>::loadu(in_ptr0 + ...)`.
# Switch to "cuda" if you want the Triton path.
device = "cpu"
a = torch.randn(64, 128, device=device, dtype=torch.float32)
b = torch.randn(64, 128, device=device, dtype=torch.float32)
c = torch.randn(64, 128, device=device, dtype=torch.float32)

compiled = torch.compile(fn, backend=backend, fullgraph=True)
out = compiled(a, b, c)
out_eager = fn(a, b, c)
torch.testing.assert_close(out, out_eager, rtol=1e-3, atol=1e-3)
print("OK, shape:", tuple(out.shape))

print(f"\nTrace dir: {TRACE_DIR}")
print(f"Cache dir: {CACHE_DIR}")
print("\nNext: open output_code.py from the cache dir and walk through")
print("04_inductor_codegen.md side-by-side. For the tlparse HTML view, run:")
print(f"  tlparse {TRACE_DIR}/dedicated_log_torch_trace_*.log --overwrite -o ./torch_trace_parsed")
