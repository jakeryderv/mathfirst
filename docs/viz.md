# Browser visualization

The first slice follows **Math → Realization → Viewer**. `FunctionGraph`
describes a function's graph and retains the source function. It is immutable
and owns no samples, rendering, or server state. `Viewer` owns numerical
evaluation, the local server, browser rendering, and its internal viewport.

Install the optional runtime with `uv add "mathfirst[viz]"` or
`pip install "mathfirst[viz]"`. From this checkout, use `uv sync --locked --extra viz`.
Core mathematical objects and imports from `mathfirst.viz` work without the extra;
starting the viewer requires it.

## Scripts

```python
from mathfirst import Polynomial, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
p = Polynomial(x**3 - 2 * x + 1, x)
viewer = Viewer(FunctionGraph(p))
viewer.show()
```

`show()` starts a server on `127.0.0.1` with an available port, opens the browser
after startup, and blocks until interrupted or `close()` is called. Ctrl+C
stops the server and returns from `show()`. Closing the browser tab does not
stop Python. Repeated calls reuse a running server; `close()` can be repeated,
and a closed viewer can be started again.

Run the complete example from the repository root:

```sh
uv run --locked --extra viz python examples/function_graph.py
```

## Practical examples

Each example keeps the math, realization, and viewer explicit:

| Example | What to observe |
| --- | --- |
| [Cubic graph](../examples/function_graph.py) | Pan, scroll zoom, hover coordinates, and reset. |
| [Bounded domain](../examples/bounded_domain.py) | The open interval `(-2, 2)` supplies the initial range; panning outside it leaves blank regions. Open endpoints have no special markers yet. |
| [Invalid values](../examples/invalid_values.py) | `sqrt(x)` is blank for negative inputs. Pan entirely into the negative range, then reset to return to the curve. |
| [Notebook viewer](../examples/notebook_viewer.ipynb) | Start explicitly in background mode, inspect the URL, and stop the server from another cell. |

Run the additional scripts from the repository root:

```sh
uv run --locked --extra viz python examples/bounded_domain.py
uv run --locked --extra viz python examples/invalid_values.py
```

Open the notebook in your notebook editor and select a local Python kernel with
the visualization extra installed. In this checkout, that is the `.venv`
environment created by `uv sync --locked --extra viz`; the kernel also needs
`ipykernel`, which your editor can install if requested. Run cells in order, keep
the kernel alive while viewing, and run the final shutdown cell when finished.
Before rerunning the cells that construct a new viewer, close the existing one.

## Notebooks and background use

Blocking behavior is identical everywhere. Choose background mode explicitly:

```python
viewer.show(block=False)
viewer.url  # Local URL, or None when stopped.
```

The call returns once the server is ready. Keep the Python process or notebook
kernel running. When finished, call:

```python
viewer.close()
```

Use `viewer.show(block=False, open_browser=False)` to start without opening a
browser. Open `viewer.url` manually if desired. Background threads are daemon
threads, and normal interpreter exit attempts cleanup; explicit `close()` is
the reliable lifecycle boundary. This viewer targets a browser on the same
computer as Python; remote notebook hosting is outside this slice.

## Behavior and limits

- One `FunctionGraph` per viewer, sourced from a one-variable `Function` or `Polynomial`.
- Domains may be unspecified, real, or a numerically evaluable interval. Other
  domain shapes are rejected for now. Interval boundaries are respected when
  sampling, including open endpoints; function evaluation itself remains unchanged.
- A bounded interval supplies the initial horizontal range. Otherwise it defaults
  to `(-5, 5)`. Override it with `Viewer(graph, xlim=(-2, 2))`.
- The browser supports pan, scroll zoom, hover coordinates, and reset. Viewport
  changes request new samples over a WebSocket. View state remains internal.
- The viewer samples 513 evenly spaced points per horizontal range using NumPy.
  Invalid, nonfinite, and complex values become gaps. This is an approximate
  graph: narrow features and poles between samples may be missed or connected.
  There is no adaptive sampling or symbolic discontinuity detection yet.
- Bounds, samples, and validity masks stay NumPy-backed inside the viewer.
  Bounds and sample arrays are owned and read-only; JSON conversion happens when
  sending WebSocket messages, with invalid samples represented as `null`.
  See [numerical contracts](numerical.md) for dtype, validation, and ownership policies.
- Functions must be supported by the NumPy evaluation backend. Initial
  evaluation failures raise a useful error before server startup; later evaluation
  errors leave the last successful graph and viewport intact.
- Plotly.js is bundled and pinned, and all assets use `importlib.resources`.
  The browser does not fetch scripts from a CDN or require an internet connection.

The runtime uses NumPy, FastAPI, Uvicorn, WebSockets, and FastAPI's Pydantic
dependency for internal message validation. These stay behind mathfirst's API.
No JavaScript build toolchain is required for this small viewer.

CLI launching, watch mode, public `ViewState`, `Figure`/export, multiple
realizations, and alternate backends remain deferred.
