# torch.compile Deep Dive

An interactive deep dive into PyTorch's `torch.compile` system, tracing the journey from Python functions to optimized FX graphs. Note: All examples run on **CPU** for simplicity purposes.

## Lessons

| # | File | Topic |
|---|------|-------|
| 1 | `01_compile_basics.py` | Compilation pipeline, debug backend, trace generation |
| 2a | `02a_static_shapes.py` | Static shape guards, recompilation on shape change |
| 2b | `02b_dynamic_shapes.py` | `dynamic=True`, `mark_dynamic` — avoiding recompilation |
| 2c | `02c_dynamic_trace.py` | Trace with dynamic shapes, symbolic dim propagation |
| 3 | `03_graph_breaks.py` | Graph breaks quiz — spot the breaks in one function |
| 3 (solution) | `03s_graph_breaks_solution.py` | Fixed version — zero graph breaks, single graph |

## Setup

```bash
pip install torch          # PyTorch 2.x
pip install tlparse        # parsing structured compilation logs
pip install pydot          # parsing graph DOT files
pip install fastapi uvicorn  # optional: compiler debug server
```

## What is torch.compile?

`torch.compile` is PyTorch 2's compiler-based optimization system introduced in PyTorch 2.0. In standard PyTorch ("eager mode"), each operation executes immediately as Python encounters it — simple to debug, but it leaves significant performance on the table because the runtime has no visibility into what comes next.

`torch.compile` changes this by capturing a **graph** of operations before executing them. This lets the compiler apply optimizations that are impossible in eager mode:

- **Operator fusion** — consecutive elementwise ops (e.g., `add -> mul -> relu`) are merged into a single kernel, eliminating intermediate memory reads/writes that dominate GPU execution time.
- **Memory planning** — the compiler sees the full graph and can allocate/reuse buffers optimally, reducing peak memory usage.
- **Backend-specific codegen** — for CUDA, the compiler generates optimized [Triton](https://triton-lang.org/) kernels with tuned tile sizes, memory coalescing, and launch configurations; for CPU, it generates C++/OpenMP code.
- **Automatic gradient graph optimization** — via AOTAutograd, both forward and backward passes are captured and optimized together, enabling cross-pass fusion and minimal activation checkpointing.

The key design principle is **zero-effort adoption**: you wrap your model or function with `torch.compile(fn)` and the compiler handles the rest. Dynamo's guard system ensures correctness — if the compiled graph's assumptions (shapes, dtypes, Python values) no longer hold, it transparently re-traces.

In practice, `torch.compile` typically delivers **10-40% training speedup** on modern GPU workloads with no code changes beyond the one-line wrapper. For inference, speedups can be even larger due to more aggressive fusion.

```python
# One-line adoption:
compiled_model = torch.compile(model)
output = compiled_model(input)  # first call traces + compiles; subsequent calls reuse cache
```

> **Recommended reading**: The [PyTorch 2 paper](https://docs.pytorch.org/assets/pytorch2-2.pdf) provides the theoretical foundation. This tutorial series focuses on hands-on exploration of the internals.

## Lesson 1: Compile Basics — Understanding the Compilation Pipeline

Let's run `01_compile_basics.py` in debug mode and understand what `torch.compile` does under the hood. The compilation pipeline has two major phases.

### The Debug Backend

`debug_backend.py` is a lightweight custom backend included in this repo that wraps TorchInductor with extra instrumentation:

1. **Structured trace export** — redirects `TORCH_TRACE` logs to a directory so every compilation stage (Dynamo bytecode analysis, AOTAutograd graph capture, Inductor IR lowering, kernel codegen) is recorded in one place.
2. **Custom graph pass hooks** — a `PassConfig` lets you inject graph-level transformations (in-place passes or `fx.Transformer` subclasses) that run before Inductor's own optimisation pipeline. When tracing is enabled, DOT files are generated showing before/after state of each pass.
3. **Inductor config validation** — the config dict is checked against `torch._inductor`'s known options at init time, catching typos early.

Under the hood, `DebugBackend.__call__` applies your custom passes, then delegates to `torch._inductor.compile_fx.compile_fx` with the validated config — so the actual code generation (Triton/C++/OpenMP) is still handled by upstream Inductor.

### Running in debug mode

`01_compile_basics.py` uses the debug backend with `torch_trace_enabled=True`, which writes structured compiler traces to `./torch_trace/`. Run it with verbose Dynamo logs:

```bash
bash run_debug.sh 01
```

This produces two complementary outputs:
- **Console logs** — verbose Dynamo/guard/recompile output (from `TORCH_LOGS`)
- **`./torch_trace/`** — structured trace files (Dynamo graphs, AOTAutograd, Inductor IR, custom pass DOT files)

### Parsing and browsing the trace logs

Use [`tlparse`](https://github.com/pytorch/tlparse) to convert the raw traces into browsable HTML:

```bash
tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
```

Open `./torch_trace_parsed/index.html` in a browser to explore:
- FX graphs at each compilation stage
- Guard details and recompilation reasons
- Custom pass before/after DOT visualizations

For a richer experience (directory browsing, inline DOT-to-SVG rendering), start the debug server:

```bash
pip install fastapi uvicorn
python compiler_debug_server.py
sudo dnf install graphviz # for dot graph view in the webserver
# Browse http://localhost:8080/view/
```

See `compiler_debug_server.py` in this directory for the server source.

### Summary of debug artifacts

| Source | File / Directory | What it shows |
|--------|-----------------|---------------|
| Console (`TORCH_LOGS`) | `+dynamo` | Bytecode tracing, FX graph construction |
| Console (`TORCH_LOGS`) | `+guards` | Guard installation and evaluation |
| Console (`TORCH_LOGS`) | `+aot` | AOTAutograd joint/forward/backward graphs |
| Debug backend trace | `./torch_trace/` | Raw structured trace (all stages) |
| tlparse output | `./torch_trace_parsed/` | Browsable HTML with graphs, guards, IR |
| `TORCH_COMPILE_DEBUG=1` | `torch_compile_debug/run_*/torchinductor/*/output_code.py` | Generated Triton/C++ kernel code |
| `TORCH_COMPILE_DEBUG=1` | `ir_pre_fusion.txt` / `ir_post_fusion.txt` | Inductor IR before/after kernel fusion |

--- 
Here is an overview of high level torch compile pipeline

### Phase 1: TorchDynamo — Graph Capture

Now let's understand what happened during compilation. The pipeline has two major phases.

TorchDynamo is PyTorch's graph capture mechanism. It hooks into CPython's frame evaluation ([PEP 523](https://peps.python.org/pep-0523/)) and rewrites bytecode to capture a computational graph instead of executing eagerly.

When `torch.compile(compute, backend=...)` is called and `compute(x, y)` runs for the first time:

1. **Bytecode analysis** — Dynamo uses `dis` to disassemble the function's bytecode, then processes each instruction through its own symbolic execution engine (`InstructionTranslator` in `symbolic_convert.py`).

2. **Symbolic execution** — Instead of running ops on real tensors, Dynamo tracks variables as `VariableTracker` objects and builds an FX graph node for each operation:
   ```
   x + y       →  call_function(aten.add.Tensor, args=(x, y))
   a * 2.0     →  call_function(aten.mul.Tensor, args=(a, 2.0))
   torch.relu  →  call_function(aten.relu.default, args=(b,))
   F.silu      →  call_function(aten.silu.default, args=(c,))
   .sum(dim=1) →  call_function(aten.sum.dim_IntList, args=(e, [1]))
   .mean()     →  call_function(aten.mean.default, args=(d,))
   ```

3. **Guard generation** — Dynamo records conditions (tensor shapes, dtypes, Python values) that must hold true for the cached graph to be valid. On subsequent calls, guards are checked in C++ for performance; if all pass, the compiled function is reused without re-tracing.

4. **Graph breaks** — When Dynamo encounters an unsupported operation (data-dependent control flow, certain Python builtins), it creates a *graph break*: the graph captured so far is compiled, and tracing restarts after the break point.

**What to look for in the debug logs** (`torch_compile_debug/run_*/torchdynamo/debug.log`):
- Bytecode tracing output for each instruction
- FX graph construction with node creation
- Guard installation for tensor shapes and types
- Graph break locations (if any)

### Phase 2: TorchInductor — Graph Optimization and Code Generation

Once Dynamo hands the FX graph to the backend, TorchInductor optimizes and lowers it into executable code. The pipeline goes through several stages:

#### AOTAutograd

If any input requires gradients, AOTAutograd captures the forward and backward graph *ahead of time* as a single joint graph, then partitions it via a min-cut algorithm to minimize tensors saved for the backward pass.

During this phase, **decompositions** are applied — high-level ops are lowered into primitives:
```python
# F.silu is decomposed into sigmoid + mul:
silu(x) → x * sigmoid(x)
```

**What to look for**: The `+aot` log token shows the joint forward+backward graph and the partitioned forward/backward subgraphs.

#### Inductor IR (Defined-by-Run IR)

The FX graph is lowered into Inductor's internal IR with explicit memory layout information. You can inspect this in:
- `ir_pre_fusion.txt` — IR nodes before kernel fusion
- `ir_post_fusion.txt` — IR nodes after fusion (fused nodes run in a single kernel)

Example IR node (from our `compute` function):
```
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: N})]
op0.sizes = ([N], [N])
class op0_loop_body:
    def body(self, ops):
        load = ops.load('arg0_1', ...)    # x
        load_1 = ops.load('arg1_1', ...)  # y
        add = ops.add(load, load_1)       # x + y
        mul = ops.mul(add, constant_2)    # * 2.0
        relu = ops.relu(mul)              # relu
        reduction = ops.reduction('sum', relu)  # sum(dim=1)
```

#### Kernel Fusion (Scheduler)

The `Scheduler` determines which IR nodes can be fused into a single kernel to minimize memory traffic:
- **Pre-fusion**: Each op is a separate node
- **Post-fusion**: Compatible ops (e.g., elementwise + reduction) are combined into `FusedSchedulerNode`s

#### Code Generation

For CPU: Inductor generates C++/OpenMP code.
For CUDA: Inductor generates Triton Python DSL kernels that are further compiled (Triton IR -> LLVM IR -> PTX -> cubin).


> **Recommended reading**: For a full code-level walkthrough of these stages, see the blog post [Torch.compile 101 — From Python Function to Triton Kernel](https://supercharleszhu.github.io/posts/00/).

## Lesson 2: Dynamic Shapes and Recompilation

By default, Dynamo specialises on the exact tensor shapes it sees during tracing. Every new shape triggers a full recompilation — expensive, and a common source of unexpected slowdowns in production. This lesson is split into three scripts you can run independently.

> **Reference**: [PyTorch Dynamic Shapes Guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html)

### 2a: The problem — static shapes cause recompilation

```bash
bash run_debug.sh 02a
```

Shows that each unique shape triggers a full re-trace + re-compile. Look for `[backend] compilation #N` in the output — you'll see 3 compilations for 3 different shapes. With `TORCH_LOGS=+recompiles,+guards` (set by `run_debug.sh`), you'll also see:

```
[recompiles] triggered because tensor 'L["x"]' size changed from (4,) to (8,)
```

### 2b: The fix — dynamic shapes

```bash
bash run_debug.sh 02b
```

Demonstrates two approaches to avoid recompilation:

- **`dynamic=True`** — all dimensions become symbolic. One compilation serves all sizes. Trade-off: slightly less optimised kernels since the compiler has less static info.
- **`mark_dynamic(tensor, dim)`** — mark only the dimensions that vary (e.g., batch). The compiler keeps static info on fixed dims. This is the **recommended** approach.

Important: call `mark_dynamic` **before** passing the tensor to the compiled function.

### 2c: Tracing dynamic shapes

```bash
bash run_debug.sh 02c
tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
```

Captures a full trace with `dynamic=True` so you can see symbolic dimensions (`s0`, `s1`, ...) propagate through the FX graph, AOTAutograd, and Inductor IR.

### What to look for in the trace

| What to check | Where to find it |
|---------------|-----------------|
| Number of compilations per function | tlparse index — multiple entries = recompilation |
| Guard conditions (static vs symbolic) | `+guards` in console logs |
| Symbolic shape names (`s0`, `s1`, ...) | FX graph in tlparse — symbolic dims replace concrete values |
| Recompilation reasons | `+recompiles` in console logs |

### When recompilation is expected

Not all recompilation is bad. Dynamo also recompiles on:
- **dtype changes** — `float32` → `float64` produces different guards
- **device changes** — CPU → CUDA
- **Python value changes** used in the graph (e.g., a flag that controls branching)

The `+guards` logs distinguish these from shape-triggered recompilation.

## Lesson 3: Graph Breaks (Quiz)

A **graph break** happens when `torch.compile` encounters code it cannot trace into the current FX graph. Dynamo compiles what it has so far, runs the unsupported code eagerly, then starts a **new** graph on the other side.

> **Reference**: [Resolving Graph Breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html#resolving-graph-breaks)

### Why graph breaks matter

- Each graph is compiled and optimised **independently** — operator fusion, memory planning, and kernel selection all stop at graph boundaries.
- More graphs = more compilation overhead at startup.
- Performance-critical code should ideally live in a single graph.

### Common causes

| Cause | Example | Fix |
|-------|---------|-----|
| Data-dependent control flow | `if x.sum() > 0:` | `torch.cond()` |
| Python builtins | `print()`, `open()` | Remove, or `@torch.compiler.disable` |
| Tensor → Python escape | `.item()`, `.tolist()` | Keep ops in tensor-land |
| Non-traceable code | Arbitrary C extensions | Wrap in custom op |

### Running the quiz

```bash
bash run_debug.sh 03
tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
```

The script has a single function `buggy_model` with **3 graph breaks** baked in. Read it, predict where the breaks are, then run to confirm.

| Break | Code pattern | Fix (in solution) |
|-------|-------------|-------------------|
| #1 | `print()` — Python builtin | Remove from compiled function |
| #2 | `.item()` — tensor → Python escape | `y - y.mean()` (stay in tensor-land) |
| #3 | `if y.sum() > 0` — data-dependent branch | `torch.cond()` (traces both branches) |

### Running the solution

```bash
bash run_debug.sh 03s
tlparse ./torch_trace/dedicated_log* -o ./torch_trace_parsed --overwrite
```

The fixed version in `03s_graph_breaks_solution.py` compiles into a **single graph** with zero breaks. Compare the tlparse output side-by-side: 4+ graph entries (buggy) vs 1 (fixed).

### What to look for in the trace

- **Multiple graph entries** in tlparse = graph breaks confirmed
- **`TORCH_LOGS=graph_breaks`** in the console shows the exact line and reason for each break
- Compare buggy (4+ graphs) vs solution (1 graph) — same logic, dramatically different compilation
