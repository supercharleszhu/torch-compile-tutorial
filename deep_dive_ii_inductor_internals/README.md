# Deep Dive II — Inductor Internals (companion code)

Hands-on companion to the blog post [**Torch.compile Deep Dive II — Inductor Codegen and Buffer Lifetimes, End to End**](https://supercharleszhu.github.io/posts/2026-04-30-00/).

This folder is for *watching* the Inductor pipeline run. The blog post explains *what* each stage does; these scripts let you see it happen on a tiny model.

## Files

| File | Purpose |
|---|---|
| `inductor_hooks.py` | Drop-in monkey-patches that print one line per Inductor stage (call_function dispatch, realize, register_buffer, scheduler `_compute_attrs`, `compute_last_usage`, wrapper `codegen_free`, Triton `codegen_kernel`). Maps 1:1 to blog stages 1–6. |
| `run_walkthrough.py` | Runs the Lesson 4 example (`(a+b)*c.relu()).sum(-1)`) under torch.compile with all hooks installed, so you can read the live pipeline trace. |

## Quickstart

```bash
cd /home/chzhu/personal/torch-compile-tutorial/deep_dive_ii_inductor_internals
python run_walkthrough.py 2>&1 | grep '\[hooks\]'
```

Filter to a single stage:

```bash
python run_walkthrough.py 2>&1 | grep 'Stage 2:'    # read_writes per node
python run_walkthrough.py 2>&1 | grep 'Stage 4:'    # last_usage decisions
python run_walkthrough.py 2>&1 | grep 'register_buffer'   # which IR types fire register_buffer
```

## Custom hooks

`inductor_hooks.install()` accepts a `stops` set if you only want a subset:

```python
import inductor_hooks
inductor_hooks.install(
    stops={"dispatch", "realize", "register_buf"},   # focus on Stage 1
    pause=True,                                       # breakpoint() at each stop
)
```

Available stops:

| Stop | Blog stage | Hooked function |
|---|---|---|
| `"dispatch"` | Stage 1 | `GraphLowering.call_function` |
| `"realize"` | Stage 1 | `TensorBox.realize` |
| `"register_buf"` | Stage 1 | `GraphLowering.register_buffer` |
| `"compute_attrs"` | Stage 2 | `SchedulerNode._compute_attrs` |
| `"last_usage"` | Stage 4 | `Scheduler.compute_last_usage` |
| `"codegen_free"` | Stage 5 | `PythonWrapperCodegen.codegen_free` |
| `"kernel"` | Stage 6 | `TritonKernel.codegen_kernel` |

## Use this with your own model

Drop the hook call at the top of any `torch.compile` script:

```python
import inductor_hooks
inductor_hooks.install()                # see everything

import torch
compiled = torch.compile(my_model)
compiled(x)                             # prints [hooks] lines as compilation runs
```

Pair with `TORCH_COMPILE_DEBUG=1` for the full picture: the hooks give you a real-time pipeline trace, the debug dump gives you the IR snapshots and final wrapper.
