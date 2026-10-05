"""Numerical sampling and the local browser runtime for one realization."""

import atexit
import importlib
import math
import socket
import threading
import time
import warnings
import webbrowser
from collections.abc import Callable
from importlib import resources
from typing import TYPE_CHECKING, Literal, Self

import sympy as sp

from .realization import FunctionGraph

if TYPE_CHECKING:
    import uvicorn
    from fastapi import FastAPI


class Viewer:
    """Explore one function graph in a local browser.

    ``show()`` blocks until interrupted or closed. Use ``show(block=False)``
    explicitly in notebooks or background applications, then call ``close()``.
    Constructing a viewer neither starts a server nor imports optional backends.
    """

    _sample_count = 513

    def __init__(
        self, graph: FunctionGraph, *, xlim: tuple[float, float] | None = None
    ) -> None:
        if not isinstance(graph, FunctionGraph):
            raise TypeError("Viewer requires a FunctionGraph")
        self._graph = graph
        source_domain = graph.function.domain
        domain = None if source_domain is None else source_domain.to_sympy()
        if (
            domain is not None
            and domain != sp.S.Reals
            and not isinstance(domain, sp.Interval)
        ):
            raise ValueError("The viewer currently supports real or interval domains")
        self._interval = domain if isinstance(domain, sp.Interval) else None
        self._domain_bounds: tuple[float, float] | None = None
        if self._interval is not None:
            try:
                self._domain_bounds = (
                    float(self._interval.start),
                    float(self._interval.end),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Interval endpoints must be numerically evaluable"
                ) from exc
        self._initial_xlim = (
            self._check_range(xlim) if xlim is not None else self._default_range()
        )
        self._xlim = self._initial_xlim
        self._ylim: tuple[float, float] | None = None
        self._evaluate: Callable[..., object] | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._url: str | None = None
        self._server_error: BaseException | None = None
        self._lifecycle_lock = threading.RLock()

    @property
    def graph(self) -> FunctionGraph:
        """The realization supplied at construction."""
        return self._graph

    @property
    def url(self) -> str | None:
        """The local URL while the server is running, otherwise None."""
        return (
            self._url if self._thread is not None and self._thread.is_alive() else None
        )

    @staticmethod
    def _check_range(bounds: tuple[float, float]) -> tuple[float, float]:
        if len(bounds) != 2 or any(isinstance(value, bool) for value in bounds):
            raise ValueError("Bounds must contain two real numbers")
        low, high = map(float, bounds)
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low >= high
            or not math.isfinite(high - low)
        ):
            raise ValueError("Bounds must be finite and increasing")
        return low, high

    def _default_range(self) -> tuple[float, float]:
        if self._domain_bounds is not None:
            low, high = self._domain_bounds
            if math.isfinite(low) and math.isfinite(high):
                return self._check_range((low, high))
        return -5.0, 5.0

    @staticmethod
    def _require_viz() -> None:
        try:
            for name in ("numpy", "fastapi", "pydantic", "uvicorn", "websockets"):
                importlib.import_module(name)
        except ImportError as exc:
            raise ImportError(
                "The browser viewer requires the visualization extra. "
                "Install 'mathfirst[viz]' or run 'uv sync --extra viz'."
            ) from exc

    def _sample(
        self, bounds: tuple[float, float]
    ) -> tuple[list[float], list[float | None]]:
        self._require_viz()
        import numpy as np

        low, high = self._check_range(bounds)
        x = np.linspace(low, high, self._sample_count)
        selected = np.ones(x.shape, dtype=bool)
        if self._interval is not None:
            assert self._domain_bounds is not None
            start, end = self._domain_bounds
            selected &= x > start if self._interval.left_open else x >= start
            selected &= x < end if self._interval.right_open else x <= end
        y: list[float | None] = [None] * len(x)
        if not selected.any():
            return x.tolist(), y

        try:
            if self._evaluate is None:
                self._evaluate = sp.lambdify(
                    self.graph.function.variables[0].to_sympy(),
                    self.graph.function.expression.to_sympy(),
                    modules="numpy",
                    dummify=True,
                )
            with np.errstate(all="ignore"):
                values = np.asarray(self._evaluate(x[selected]), dtype=np.complex128)
                values = np.broadcast_to(values, x[selected].shape)
            valid = (
                np.isfinite(values.real) & np.isfinite(values.imag) & (values.imag == 0)
            )
            for index, value, usable in zip(
                np.flatnonzero(selected), values.real, valid, strict=True
            ):
                if usable:
                    y[int(index)] = float(value)
        except (
            TypeError,
            ValueError,
            NameError,
            ZeroDivisionError,
            OverflowError,
            NotImplementedError,
        ) as exc:
            raise ValueError(
                "This function cannot be sampled with the NumPy backend"
            ) from exc
        return x.tolist(), y

    @staticmethod
    def _asset(name: str) -> bytes:
        return (
            resources.files("mathfirst.viz")
            .joinpath("static", *name.split("/"))
            .read_bytes()
        )

    def _create_app(self) -> "FastAPI":
        self._require_viz()
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import HTMLResponse, Response
        from pydantic import (
            BaseModel,
            ConfigDict,
            Field,
            ValidationError,
            model_validator,
        )
        from starlette.concurrency import run_in_threadpool

        viewer = self

        class Viewport(BaseModel):
            model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
            type: Literal["viewport"]
            request_id: int = Field(ge=0, le=2**53 - 1)
            xlim: tuple[float, float]
            ylim: tuple[float, float] | None = None

            @model_validator(mode="after")
            def check_bounds(self) -> Self:
                viewer._check_range(self.xlim)
                if self.ylim is not None:
                    viewer._check_range(self.ylim)
                return self

        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/")
        async def index() -> HTMLResponse:
            return HTMLResponse(viewer._asset("index.html"))

        @app.get("/assets/{name:path}")
        async def asset(name: str) -> Response:
            assets = {
                "viewer.js": ("viewer.js", "application/javascript"),
                "viewer.css": ("viewer.css", "text/css"),
                "plotly.min.js": ("vendor/plotly.min.js", "application/javascript"),
            }
            if name not in assets:
                raise HTTPException(status_code=404)
            path, media_type = assets[name]
            return Response(viewer._asset(path), media_type=media_type)

        @app.websocket("/ws")
        async def updates(websocket: WebSocket) -> None:
            origin = websocket.headers.get("origin")
            if (
                origin is not None
                and origin != f"http://{websocket.headers.get('host')}"
            ):
                await websocket.close(code=1008)
                return
            await websocket.accept()
            await websocket.send_json(
                {
                    "type": "ready",
                    "label": str(viewer.graph.function.expression),
                    "variable": viewer.graph.function.variables[0].name,
                    "xlim": viewer._xlim,
                    "ylim": viewer._ylim,
                    "initial_xlim": viewer._initial_xlim,
                }
            )
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    try:
                        request = Viewport.model_validate_json(
                            message.get("text") or message.get("bytes") or b""
                        )
                    except ValidationError:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "request_id": None,
                                "message": "Invalid viewport: bounds must be finite and increasing.",
                            }
                        )
                        continue
                    try:
                        x, y = await run_in_threadpool(viewer._sample, request.xlim)
                    except ValueError as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "request_id": request.request_id,
                                "message": str(exc),
                            }
                        )
                        continue
                    viewer._xlim = request.xlim
                    viewer._ylim = request.ylim
                    await websocket.send_json(
                        {
                            "type": "samples",
                            "request_id": request.request_id,
                            "x": x,
                            "y": y,
                        }
                    )
            except WebSocketDisconnect:
                return

        return app

    def _start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._server is not None and self._server.should_exit:
                    raise RuntimeError("The viewer is shutting down")
                return
            self._require_viz()
            import uvicorn

            # Check evaluation and package resources before starting a thread.
            self._sample(self._xlim)
            app = self._create_app()
            self._asset("index.html")
            self._asset("vendor/plotly.min.js")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", 0))
                self._socket = sock
                self._server_error = None
                self._server = uvicorn.Server(
                    uvicorn.Config(
                        app,
                        host="127.0.0.1",
                        loop="asyncio",
                        ws="websockets-sansio",
                        access_log=False,
                        log_level="warning",
                        log_config=None,
                        timeout_graceful_shutdown=1,
                        ws_max_size=16384,
                    )
                )
                server = self._server

                def serve() -> None:
                    try:
                        server.run(sockets=[sock])
                    except BaseException as exc:
                        self._server_error = exc

                self._thread = threading.Thread(
                    target=serve, name="mathfirst-viewer", daemon=True
                )
                self._thread.start()
                deadline = time.monotonic() + 10
                while not server.started:
                    if not self._thread.is_alive():
                        raise RuntimeError(
                            "The viewer server failed to start"
                        ) from self._server_error
                    if time.monotonic() >= deadline:
                        raise RuntimeError("The viewer server did not become ready")
                    time.sleep(0.01)
                self._url = f"http://127.0.0.1:{sock.getsockname()[1]}/"
                atexit.register(self.close)
            except BaseException:
                self.close()
                sock.close()
                raise

    def show(self, *, block: bool = True, open_browser: bool = True) -> Self:
        """Start or reuse the server and optionally open its URL.

        Blocking behavior is identical in scripts and notebooks. With
        ``block=False``, this returns immediately after startup; the Python
        process must remain alive and the caller is responsible for ``close()``.
        """
        if not isinstance(block, bool) or not isinstance(open_browser, bool):
            raise TypeError("block and open_browser must be booleans")
        self._start()
        url = self.url
        assert url is not None
        if open_browser and not webbrowser.open(url):
            warnings.warn(
                f"Could not open a browser; open {url} manually.",
                RuntimeWarning,
                stacklevel=2,
            )
        if block:
            try:
                thread = self._thread
                assert thread is not None
                # Avoid interrupting Thread.join(), which can mark a live
                # thread as stopped on Python 3.12.
                while thread.is_alive():
                    time.sleep(0.05)
                if self._server_error is not None:
                    raise RuntimeError(
                        "The viewer server stopped unexpectedly"
                    ) from self._server_error
            except KeyboardInterrupt:
                pass
            finally:
                self.close()
        return self

    def close(self) -> None:
        """Stop the server and release its socket; safe to call repeatedly."""
        with self._lifecycle_lock:
            if self._server is not None:
                self._server.should_exit = True
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(timeout=5)
                if self._thread.is_alive():
                    raise RuntimeError("The viewer server did not stop")
            if self._socket is not None:
                self._socket.close()
            self._server = None
            self._thread = None
            self._socket = None
            self._url = None
            atexit.unregister(self.close)
