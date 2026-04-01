"""
Compiler Debug Service

A lightweight FastAPI server for browsing parsed torch.compile trace output.
Serves files from ./torch_trace_parsed with directory listing, inline HTML
rendering, and automatic DOT-to-SVG conversion via GraphViz.

Usage:
    pip install fastapi uvicorn
    python compiler_debug_server.py
    # Browse http://localhost:8080/view/
"""

import html as html_mod
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
import uvicorn

app = FastAPI(title="Compiler Debug Service")

SERVE_DIR = Path(
    os.getenv("SERVE_DIR", "./")
).resolve()


@app.get("/")
async def root():
    """Health check and usage info."""
    return {
        "service": "compiler-debug-service",
        "serve_dir": str(SERVE_DIR),
        "endpoints": {"/view/<path>": "Browse files and directories"},
    }


@app.get("/view/{file_path:path}")
async def view_file(file_path: str):
    """Browse directories and serve files from SERVE_DIR."""
    full_path = (SERVE_DIR / file_path).resolve()

    # Security: ensure resolved path stays under SERVE_DIR
    if not full_path.is_relative_to(SERVE_DIR):
        raise HTTPException(403, "Path traversal detected")

    if not full_path.exists():
        raise HTTPException(404, f"Not found: {file_path}")

    # --- Directory listing ---
    if full_path.is_dir():
        items = sorted(full_path.iterdir())
        links = []
        for item in items:
            rel = str(item.relative_to(SERVE_DIR))
            icon = "📁" if item.is_dir() else "📄"
            safe_name = html_mod.escape(item.name, quote=True)
            safe_rel = html_mod.escape(rel, quote=True)
            links.append(
                f'<div style="padding:4px 0">'
                f'<a href="/view/{safe_rel}">{icon} {safe_name}</a>'
                f"</div>"
            )
        safe_fp = html_mod.escape(file_path, quote=True)
        body = "\n".join(links) if links else "<p>(empty directory)</p>"
        return HTMLResponse(
            f"<html><head><title>/{safe_fp}</title>"
            f'<style>body{{font-family:sans-serif;margin:20px}}'
            f"a{{text-decoration:none}}a:hover{{text-decoration:underline}}</style>"
            f"</head><body><h1>/{safe_fp}</h1>{body}</body></html>"
        )

    # --- File serving ---
    suffix = full_path.suffix.lower()

    if suffix in (".html", ".htm"):
        return FileResponse(full_path, media_type="text/html")

    if suffix in (".txt", ".log"):
        return FileResponse(full_path, media_type="text/plain; charset=utf-8")

    if suffix == ".json":
        return FileResponse(full_path, media_type="application/json")

    if suffix == ".svg":
        return FileResponse(full_path, media_type="image/svg+xml")

    if suffix == ".dot":
        # Try graphviz CLI first, fall back to pydot if dot binary is missing
        try:
            result = subprocess.run(
                ["dot", "-Tsvg", str(full_path)],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return Response(content=result.stdout, media_type="image/svg+xml")
            raise HTTPException(
                500, f"GraphViz failed: {result.stderr.decode()}"
            )
        except FileNotFoundError:
            # dot binary not installed — try pydot as fallback
            try:
                import pydot

                graphs = pydot.graph_from_dot_file(str(full_path))
                if graphs:
                    svg_bytes = graphs[0].create_svg()
                    return Response(content=svg_bytes, media_type="image/svg+xml")
                raise HTTPException(500, "pydot failed to parse .dot file")
            except ImportError:
                raise HTTPException(
                    500,
                    "Cannot render .dot files: install graphviz (system package) "
                    "or pydot (pip install pydot)",
                )

    # Fallback: download
    return FileResponse(full_path, filename=full_path.name)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Serving files from {SERVE_DIR} on http://localhost:{port}/view/")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
