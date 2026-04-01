"""
Debug Backend — a lightweight torch.compile backend for learning and debugging.

Wraps TorchInductor with:
  1. Structured TORCH_TRACE export to a directory (via torch's internal logging)
  2. Custom graph-pass hooks with optional DOT-file visualization (before/after)
  3. Inductor config validation

Usage:
    from debug_backend import DebugBackend, DebugCompilationConfig, PassConfig

    config = DebugCompilationConfig(
        inductor_config={"max_autotune": True},
        pass_config=PassConfig(graph_pass=my_pass),
        torch_trace_enabled=True,
        torch_trace_dir="./torch_trace",
    )
    backend = DebugBackend(compilation_config=config)
    compiled = torch.compile(fn, backend=backend)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, NamedTuple, Sequence

import torch
import torch.fx as fx
import torch._logging._internal as logging_internal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class PassConfig(NamedTuple):
    """Configuration for custom graph passes.

    Args:
        graph_pass: Callable that takes a GraphModule and modifies it in-place.
        transformer_classes: Dict mapping names to ``fx.Transformer`` subclasses.
            Each transformer is instantiated and ``.transform()`` is called to
            produce a new graph.  Applied sequentially after ``graph_pass``.
    """
    graph_pass: Callable[[fx.GraphModule], None] | None = None
    transformer_classes: dict[str, type] | None = None


class DebugCompilationConfig(NamedTuple):
    """Configuration for :class:`DebugBackend`.

    Args:
        inductor_config: Dict of TorchInductor options
            (see ``torch._inductor.list_options()``).
        pass_config: Custom pass configuration.
        torch_trace_dir: Directory for structured trace output.
        torch_trace_enabled: Whether to redirect TORCH_TRACE logs.
        write_visualized_graph: Write Inductor's own graph-transform DOT files.
        eager: Skip Inductor and return the FX graph for eager execution.
    """
    inductor_config: dict[str, Any] = {}
    pass_config: PassConfig = PassConfig()
    torch_trace_dir: str = ""
    torch_trace_enabled: bool = False
    write_visualized_graph: bool = False
    eager: bool = False


# ---------------------------------------------------------------------------
# Graph-transform observer (simplified, self-contained)
# ---------------------------------------------------------------------------

class _GraphTransformObserver:
    """Context manager that records DOT files showing graph state before/after a pass.

    Only activates when *log_url* is not None.  Uses ``FxGraphDrawer`` from
    ``torch.fx.passes.graph_drawer`` (ships with PyTorch).
    """

    _pass_count: int = 0

    def __init__(self, gm: fx.GraphModule, passname: str, log_url: str | None = None):
        self.gm = gm
        self.passname = passname
        self.log_url = log_url
        self.active = log_url is not None
        if self.active:
            _GraphTransformObserver._pass_count += 1
            from torch.fx.passes.graph_drawer import FxGraphDrawer
            self._drawer_cls = FxGraphDrawer
            self._input_dot = FxGraphDrawer(
                gm, passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            self._erased: set[str] = set()
            self._created: set[str] = set()

    # -- hooks ---------------------------------------------------------------
    def _on_create(self, node: fx.Node) -> None:
        self._created.add(node.name)

    def _on_erase(self, node: fx.Node) -> None:
        self._erased.add(node.name)

    # -- context manager -----------------------------------------------------
    def __enter__(self):
        if self.active:
            self.gm._register_create_node_hook(self._on_create)
            self.gm._register_erase_node_hook(self._on_erase)
        return self

    def __exit__(self, *exc):
        if not self.active:
            return
        self.gm._unregister_create_node_hook(self._on_create)
        self.gm._unregister_erase_node_hook(self._on_erase)

        pc = _GraphTransformObserver._pass_count
        # Colour erased nodes yellow in the *input* graph
        for e in self._input_dot.get_node_list():
            e.obj_dict["attributes"]["fillcolor"] = (
                "yellow" if e.get_name() in self._erased else "grey"
            )
        self._input_dot.write(
            os.path.join(self.log_url, f"pass_{pc}_{self.passname}_input_graph.dot")
        )

        # Colour created nodes yellow in the *output* graph
        output_dot = self._drawer_cls(
            self.gm, self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()
        for e in output_dot.get_node_list():
            e.obj_dict["attributes"]["fillcolor"] = (
                "yellow" if e.get_name() in self._created else "grey"
            )
        output_dot.write(
            os.path.join(self.log_url, f"pass_{pc}_{self.passname}_output_graph.dot")
        )


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

class DebugBackend:
    """A debug-friendly torch.compile backend that wraps TorchInductor.

    Features:
      - Redirects TORCH_TRACE structured logs to a directory.
      - Runs user-supplied graph passes (in-place and/or Transformer-based)
        with optional DOT-file visualisation of before/after state.
      - Validates inductor config keys at init time.
      - Delegates code generation to ``torch._inductor.compile_fx``.

    Example::

        def my_pass(gm):
            gm.graph.lint()
            gm.recompile()

        config = DebugCompilationConfig(
            inductor_config={"max_autotune": True},
            pass_config=PassConfig(graph_pass=my_pass),
            torch_trace_enabled=True,
            torch_trace_dir="./torch_trace",
        )
        backend = DebugBackend(compilation_config=config)
        compiled = torch.compile(fn, backend=backend)
    """

    compiler_name = "debug_backend"

    def __init__(self, compilation_config: DebugCompilationConfig | None = None):
        if compilation_config is None:
            compilation_config = DebugCompilationConfig()

        self.torch_trace_enabled = False
        self.torch_trace_dir = compilation_config.torch_trace_dir

        if compilation_config.torch_trace_enabled:
            self.torch_trace_enabled = True
            if not self.torch_trace_dir:
                self.torch_trace_dir = "./torch_trace"
                logger.warning("No torch_trace_dir set — defaulting to %s", self.torch_trace_dir)
            self._setup_torch_trace_logging(self.torch_trace_dir)
            logger.info("TORCH_TRACE logging enabled → %s", self.torch_trace_dir)

        inductor_config = dict(compilation_config.inductor_config)
        if compilation_config.write_visualized_graph:
            inductor_config["trace.log_url_for_graph_xform"] = self.torch_trace_dir

        self.inductor_config = self._validate_inductor_config(inductor_config)
        self.pass_config = compilation_config.pass_config
        self.eager = compilation_config.eager

    # -- TORCH_TRACE setup ---------------------------------------------------

    @staticmethod
    def _setup_torch_trace_logging(log_directory: str) -> None:
        trace_log = logging.getLogger("torch.__trace")
        for handler in trace_log.handlers[:]:
            trace_log.removeHandler(handler)
        new_handler = logging_internal.LazyTraceHandler(log_directory)
        new_handler.setFormatter(logging_internal.TorchLogsFormatter(trace=True))
        trace_log.addHandler(new_handler)
        logging_internal.LOG_TRACE_HANDLER = new_handler

    # -- Config validation ---------------------------------------------------

    @staticmethod
    def _validate_inductor_config(options: dict[str, Any]) -> dict[str, Any]:
        from torch._inductor import config
        from typing_extensions import get_origin

        current = config.get_config_copy()
        validated: dict[str, Any] = {}
        for key, val in options.items():
            attr = key.replace("-", "_")
            if attr not in current:
                raise RuntimeError(
                    f"Unknown inductor option '{key}'. "
                    f"Run torch._inductor.list_options() to see valid options."
                )
            attr_type = config.get_type(attr)
            if get_origin(attr_type) is None and not isinstance(val, attr_type) and attr_type is not type(None):
                raise RuntimeError(
                    f"Type mismatch for '{key}': got {type(val).__name__}, "
                    f"expected {type(current[attr]).__name__}"
                )
            validated[attr] = val
        return validated

    # -- Compilation entry point ---------------------------------------------

    def __call__(self, graph: fx.GraphModule, example_inputs: Sequence[Any]) -> Callable:
        from torch._inductor.compile_fx import compile_fx

        log_url = self.torch_trace_dir if self.torch_trace_enabled else None

        # Step 1: in-place graph pass
        if self.pass_config.graph_pass is not None:
            logger.info("Applying custom graph pass (in-place)")
            with _GraphTransformObserver(graph, "custom_pass", log_url=log_url):
                self.pass_config.graph_pass(graph)

        # Step 2: Transformer-based passes
        if self.pass_config.transformer_classes is not None:
            for name, cls in self.pass_config.transformer_classes.items():
                logger.info("Applying transformer '%s': %s", name, cls.__name__)
                transformer = cls(graph)
                graph = transformer.transform()

        if self.eager:
            logger.info("Eager mode — skipping Inductor compilation")
            return graph

        return compile_fx(graph, example_inputs, config_patches=self.inductor_config)

    def __repr__(self) -> str:
        return (
            f"DebugBackend(inductor_config={self.inductor_config}, "
            f"pass_config={self.pass_config})"
        )
