"""Drop-in instrumentation for Inductor internals discussed in
"Torch.compile Deep Dive II — Inductor Codegen and Buffer Lifetimes".

Usage:
    import inductor_hooks
    inductor_hooks.install()        # all hooks, no pdb pause

    # or selectively:
    inductor_hooks.install(stops={"dispatch", "realize", "last_usage"})
    inductor_hooks.install(pause=True)   # pdb breakpoint at each stop

Stops:
  "dispatch"     — every call_function lowering (Stage 1)
  "realize"      — every TensorBox.realize() call site (Stage 1)
  "register_buf" — every V.graph.register_buffer call (Stage 1)
  "compute_attrs" — SchedulerNode._compute_attrs, the read_writes fork (Stage 2)
  "last_usage"   — Scheduler.compute_last_usage, the reverse walk (Stage 4)
  "codegen_free" — wrapper.codegen_free emitting `del` lines (Stage 5)
  "kernel"       — TritonKernel.codegen_kernel final string (Stage 6)
"""
from __future__ import annotations
import functools

_installed = False


def _bar(title: str) -> None:
    print(f"\n[hooks] {title}", flush=True)


def install(stops: set[str] | None = None, pause: bool = False) -> None:
    """Patch all (or a subset of) hook points. Idempotent."""
    global _installed
    if _installed:
        return
    _installed = True

    if stops is None:
        stops = {
            "dispatch", "realize", "register_buf",
            "compute_attrs", "last_usage", "codegen_free", "kernel",
        }

    # Stage 1 — every call_function dispatch.
    if "dispatch" in stops:
        from torch._inductor import graph as _g

        _orig = _g.GraphLowering.call_function

        @functools.wraps(_orig)
        def patched(self, target, args, kwargs):
            target_name = getattr(target, "__name__", repr(target))
            _bar(f"Stage 1: call_function {target_name}")
            ret = _orig(self, target, args, kwargs)
            ret_kind = type(ret).__name__ if hasattr(ret, "__class__") else type(ret).__name__
            if hasattr(ret, "data"):
                inner = type(ret.data).__name__
                _bar(f"        -> {ret_kind}({inner})")
            else:
                _bar(f"        -> {ret_kind}")
            if pause:
                breakpoint()
            return ret

        _g.GraphLowering.call_function = patched

    # Stage 1 — every realize() call.
    if "realize" in stops:
        from torch._inductor import ir as _ir

        _orig = _ir.TensorBox.realize

        @functools.wraps(_orig)
        def patched(self):
            inner = type(self.data).__name__
            ret = _orig(self)
            new_inner = type(self.data).__name__
            if inner != new_inner:
                _bar(f"realize() {inner} -> {new_inner}  name={ret}")
            else:
                _bar(f"realize() noop ({inner})  name={ret}")
            if pause:
                breakpoint()
            return ret

        _ir.TensorBox.realize = patched

    # Stage 1 — every register_buffer.
    if "register_buf" in stops:
        from torch._inductor import graph as _g

        _orig = _g.GraphLowering.register_buffer

        @functools.wraps(_orig)
        def patched(self, buffer, *, set_name=False):
            kind = type(buffer).__name__
            ret = _orig(self, buffer, set_name=set_name)
            _bar(f"register_buffer  {ret}  ({kind})")
            return ret

        _g.GraphLowering.register_buffer = patched

    # Stage 2 — read_writes fork.
    if "compute_attrs" in stops:
        from torch._inductor import scheduler as _sched
        from torch._inductor import ir as _ir

        _orig = _sched.SchedulerNode._compute_attrs

        @functools.wraps(_orig)
        def patched(self, *a, **kw):
            ret = _orig(self, *a, **kw)
            cls = type(self.node).__name__
            tag = "Path B (template)" if isinstance(self.node, _ir.TemplateBuffer) else "Path A (loop body)"
            reads = sorted(d.name for d in self.read_writes.reads)
            writes = sorted(d.name for d in self.read_writes.writes)
            _bar(f"Stage 2: SchedulerNode {self.get_name()} ({cls})  {tag}")
            _bar(f"         reads  = {reads}")
            _bar(f"         writes = {writes}")
            if pause:
                breakpoint()
            return ret

        _sched.SchedulerNode._compute_attrs = patched

    # Stage 4 — compute_last_usage reverse walk.
    if "last_usage" in stops:
        from torch._inductor import scheduler as _sched

        _orig = _sched.Scheduler.compute_last_usage

        @functools.wraps(_orig)
        def patched(self):
            _bar("Stage 4: compute_last_usage (reverse walk)")
            ret = _orig(self)
            for n in self.nodes:
                if n.last_usage:
                    _bar(f"   {n.get_name()}  -> del {sorted(n.last_usage)}")
            if pause:
                breakpoint()
            return ret

        _sched.Scheduler.compute_last_usage = patched

    # Stage 5 — codegen_free.
    if "codegen_free" in stops:
        from torch._inductor.codegen import wrapper as _wrap
        from torch._inductor import ir as _ir

        _orig = _wrap.PythonWrapperCodegen.codegen_free

        @functools.wraps(_orig)
        def patched(self, buffer):
            name = buffer.get_name()
            kind = "FreeLine (always)" if isinstance(buffer, (_ir.InputBuffer, _ir.TorchBindObject)) else "FreeIfNotReusedLine"
            _bar(f"Stage 5: codegen_free  {name}  ({kind})")
            return _orig(self, buffer)

        _wrap.PythonWrapperCodegen.codegen_free = patched

    # Stage 6 — final TritonKernel string.
    if "kernel" in stops:
        from torch._inductor.codegen import triton as _tri

        _orig = _tri.TritonKernel.codegen_kernel

        @functools.wraps(_orig)
        def patched(self, name=None):
            ret = _orig(self, name=name)
            n_lines = len(ret.splitlines())
            _bar(f"Stage 6: codegen_kernel  {name or '<anon>'}  ({n_lines} lines)")
            if pause:
                breakpoint()
            return ret

        _tri.TritonKernel.codegen_kernel = patched

    print(f"[inductor_hooks] installed: {sorted(stops)}  (pause={pause})", flush=True)
