# Lesson 4: A Worked Example — From `(a+b)*c.relu()).sum(-1)` to a Single Fused Kernel

A hands-on companion to [Torch.compile Deep Dive II](https://supercharleszhu.github.io/posts/2026-04-30-00). This walks one tiny function through all six stages of Inductor's pipeline (lowering, scheduler, fusion, memory planning, wrapper codegen, kernel codegen) so you can see — in actual generated code — what every stage produces.

> Anchor commit: [`e2f06e4`](https://github.com/pytorch/pytorch/tree/e2f06e4e10823421ca984267f673f45e24018856).

## Setup

```bash
cd /home/chzhu/personal/torch-compile-tutorial
python 04_inductor_codegen.py
# Optional: render the structured trace as HTML
tlparse ./torch_trace/dedicated_log_torch_trace_*.log --overwrite -o ./torch_trace_parsed
```

The script prints the path to its generated `output_code.py` after running. Open that file alongside this doc — every stage below maps to specific lines in it.

## The function

```python
def fn(a, b, c):
    return ((a + b) * c.relu()).sum(dim=-1)
```

Three pointwise ops (`add`, `relu`, `mul`) + one reduction (`sum`), all on tensors of shape `(64, 128)`. Three graph inputs. One graph output.

## Stage 1 — Lowering: FX → Inductor IR

After Dynamo + AOTAutograd, Inductor receives an FX graph with three placeholders and four `call_function` nodes. As `GraphLowering` walks them:

* `arg0_1`, `arg1_1`, `arg2_1` get registered as `InputBuffer`s — these are the placeholder names from FX, and they're load-bearing strings every downstream layer references.
* Each `call_function(aten.<op>, ...)` runs through [`lowerings`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/lowering.py#L116). The pointwise ops produce `Pointwise` IR nodes; the reduction produces a `Reduction`. Both materialize as `ComputedBuffer`s with monotonically-numbered `bufN` names.
* The intermediate `bufN`s for `add`, `relu`, `mul` get names but, because they all flow into the reduction with no other consumers, **they will not survive into the wrapper as actual allocations** — Stage 3 will fuse them away.
* The reduction's output `bufN` becomes the function output and is pinned via `V.graph.get_output_names()`.

## Stage 2 — Scheduler: `read_writes` per node

Each IR buffer becomes a `SchedulerNode`. For our `ComputedBuffer`s, [`SchedulerNode._compute_attrs`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/scheduler.py#L1554) takes the **else** branch (Path A — symbolic interpretation of the loop body):

```python
self.set_read_writes(
    dependencies.extract_read_writes(self._body, *self._sizes, normalize=should_normalize)
)
```

The result, after the four nodes have been built:

| Node | `reads` | `writes` |
|---|---|---|
| `add` | `arg0_1`, `arg1_1` | `buf_add` |
| `relu` | `arg2_1` | `buf_relu` |
| `mul` | `buf_add`, `buf_relu` | `buf_mul` |
| `sum` | `buf_mul` | `buf0` (output) |

Each entry comes from a virtualized `ops.load` or `ops.store` call inside the corresponding loop body. **No captured tensors, no template inputs — every read flows from the loop body itself.**

## Stage 3 — Fusion

The fusion pass collapses adjacent compatible nodes. For our four nodes, all three vertical fusion candidates pass:

1. **Legality** — [`SIMDScheduling.can_fuse`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/simd.py#L1308): same iteration domain `(64, 128)`, no split-scan/reduction conflicts. ✓
2. **Profitability** — [`score_fusion_memory`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/scheduler.py#L6455): every adjacent pair shares an intermediate buffer (`buf_add`, `buf_relu`, `buf_mul`), so the bytes-saved score is high. Sorted to the top of the worklist.
3. **Cycle safety** — [`will_fusion_create_cycle`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/scheduler.py#L5204): linear chain, no cycles introduced.

The result: one [`FusedSchedulerNode`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/scheduler.py#L1962) wrapping all four ops. Its `read_writes` is the union of the four child nodes' reads/writes minus internally-satisfied deps:

| Fused node | `reads` | `writes` |
|---|---|---|
| `add → relu → mul → sum` | `arg0_1`, `arg1_1`, `arg2_1` | `buf0` |

The intermediates `buf_add`, `buf_relu`, `buf_mul` disappear from the public read/write set — they exist only inside the fused kernel body, register-resident.

## Stage 4 — Memory planning + last-use

[`Scheduler.compute_last_usage`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/scheduler.py#L6811) walks in reverse:

1. Seed: `future_used_buffers = {buf0}` (the graph output).
2. Process the fused node (in reverse, it's the last node):
   * `used = {arg0_1, arg1_1, arg2_1, buf0}`
   * `last_usage = used - future_used = {arg0_1, arg1_1, arg2_1}`
   * Update `future_used = {arg0_1, arg1_1, arg2_1, buf0}`

`buf0` is in `future_used_buffers` from the seed, so no node ever claims it as `last_usage` → it never gets `del`'d. The three graph inputs get claimed by the fused node.

## Stage 5 — Wrapper codegen

`PythonWrapperCodegen` walks in execution order and emits one node:

```python
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64,), (1,), torch.float32)
        triton_red_fused_add_mul_relu_sum_0.run(
            arg0_1, arg1_1, arg2_1, buf0, 64, 128, ...)
        del arg0_1
        del arg1_1
        del arg2_1
        return (buf0,)
```

Map every line to the previous stages:

| Line | Source |
|---|---|
| `arg*_1 = args` | Stage 1's `placeholder` registration |
| `buf0 = empty_strided_cuda(...)` | Stage 2's allocation request for the reduction output |
| `triton_red_fused_*.run(arg0_1, arg1_1, arg2_1, buf0, ...)` | Stage 3's fused node + Stage 6's kernel signature |
| `del arg0_1; del arg1_1; del arg2_1` | Stage 4's `last_usage` for the fused node, emitted via [`codegen_free`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/wrapper.py#L3691)'s `FreeLine` branch (graph inputs always emit `del`) |
| `return (buf0,)` | Stage 1's `output` registration |

Three `del`s, three live-through-the-whole-kernel inputs. Zero intermediate allocations because Stage 3 fused them away.

## Stage 6 — Triton kernel codegen

[`SIMDScheduling.codegen_node`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/simd.py#L1841) builds a [`TritonKernel`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/triton.py#L2789) and walks the fused node's body via the **virtualized ops pattern** — every `ops.load`/`ops.store` in the IR's `inner_fn` dispatches to `TritonOverrides`, emitting `tl.load`/`tl.store` strings.

The output:

```python
@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={...},
)
@triton.jit
def triton_red_fused_add_mul_relu_sum_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0,
                                         xnumel, rnumel, XBLOCK, RBLOCK):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    x0 = xindex
    _tmp4 = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + tl.arange(0, RBLOCK)
        rmask = rindex < rnumel
        tmp0 = tl.load(in_ptr0 + (rindex + rnumel * x0), rmask & xmask)
        tmp1 = tl.load(in_ptr1 + (rindex + rnumel * x0), rmask & xmask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.load(in_ptr2 + (rindex + rnumel * x0), rmask & xmask)
        tmp_relu = triton_helpers.maximum(tmp3, 0.0)
        tmp4 = tmp2 * tmp_relu
        _tmp4 = _tmp4 + tmp4
    tmp4 = tl.sum(_tmp4, 1)
    tl.store(out_ptr0 + x0, tmp4, xmask)
```

| Element | Where it came from |
|---|---|
| `in_ptr0, in_ptr1, in_ptr2` | Three reads in Stage 2's `read_writes.reads` |
| `out_ptr0` | One write in Stage 2's `read_writes.writes` |
| `XBLOCK`, `RBLOCK` | [`select_tiling`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/simd.py#L3014) decision |
| `rindex + rnumel * x0` | [`codegen_indexing`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/simd.py#L1056) rewrite of the original sympy index |
| `xmask`, `rmask` | Tail-block masks paired with each index |
| The reduction loop structure | [`codegen_body`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/triton.py#L5181) for `Reduction` IR nodes |
| `tmp0 + tmp1`, `triton_helpers.maximum(...)` | [`TritonOverrides.add` / `.relu`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/triton.py#L1177) |
| `tmp2 = tmp0 + tmp1` (one line, no re-load) | [`CSE`](https://github.com/pytorch/pytorch/blob/e2f06e4e10823421ca984267f673f45e24018856/torch/_inductor/codegen/common.py#L1952) deduping intermediates |

The whole kernel is one fused body; what was four ATen ops in the source becomes a single GPU launch with no intermediate HBM traffic for `buf_add`, `buf_relu`, or `buf_mul`.

## Try it yourself

```bash
python 04_inductor_codegen.py
```

The script:

1. Compiles `fn` under the lesson's `DebugBackend` (so the trace lands in `./torch_trace/`).
2. Calls it once, which triggers Inductor compilation.
3. Prints the trace dir and inductor cache dir.

Read the wrapper line by line and map each to a stage above. The mapping should be exact.
