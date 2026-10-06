"""HTTP resources, websocket recovery, and real server lifecycle."""

import json
import threading
import time
from urllib.request import urlopen

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.sync.client import connect

from mathfirst import Function, Polynomial, Variable
from mathfirst.viz import FunctionGraph, Viewer


@pytest.fixture
def viewer():
    x = Variable("x")
    instance = Viewer(FunctionGraph(Polynomial(x**3 - 2 * x + 1, x)))
    yield instance
    instance.close()


def viewport(request_id=1, **changes):
    return {"type": "viewport", "request_id": request_id, "xlim": [-1, 1], **changes}


def test_packaged_assets_are_served_locally(viewer):
    with TestClient(viewer._create_app()) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "/assets/plotly.min.js" in page.text
        assert "https://" not in page.text
        for name, content_type in (
            ("viewer.css", "text/css"),
            ("viewer.js", "application/javascript"),
            ("plotly.min.js", "application/javascript"),
        ):
            response = client.get(f"/assets/{name}")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(content_type)
            assert response.content
        assert "v4.1.2" in client.get("/assets/plotly.min.js").text[:500]
        for path in (
            "/assets/unknown",
            "/assets/vendor/LICENSE.plotly",
            "/assets/%2e%2e/pyproject.toml",
            "/docs",
        ):
            assert client.get(path).status_code == 404


def test_viewport_resamples_without_changing_realization(viewer):
    graph = viewer.graph
    with TestClient(viewer._create_app()) as client:
        with client.websocket_connect("/ws") as websocket:
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            assert ready["variable"] == "x"
            assert ready["xlim"] == [-5, 5]
            for request_id, bounds in ((1, [-1, 1]), (2, [2, 4])):
                websocket.send_json(viewport(request_id, xlim=bounds, ylim=[-10, 10]))
                samples = websocket.receive_json()
                assert samples["type"] == "samples"
                assert samples["request_id"] == request_id
                assert samples["x"][0] == bounds[0]
                assert samples["x"][-1] == bounds[1]
                assert samples["y"] == pytest.approx(
                    [x**3 - 2 * x + 1 for x in samples["x"]]
                )
        with client.websocket_connect("/ws") as websocket:
            ready = websocket.receive_json()
            assert ready["xlim"] == [2, 4]
            assert ready["ylim"] == [-10, 10]
            assert ready["initial_xlim"] == [-5, 5]
    assert viewer.graph is graph
    assert viewer._xlim.dtype == viewer._ylim.dtype == np.dtype(np.float64)
    assert not viewer._xlim.flags.writeable and not viewer._ylim.flags.writeable


def test_numpy_nonfinite_samples_are_serialized_as_json_null():
    x = Variable("x")
    viewer = Viewer(FunctionGraph(Function(1 / x, (x,))))
    with (
        TestClient(viewer._create_app()) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_json(viewport())
        text = websocket.receive_text()
        assert "NaN" not in text and "Infinity" not in text
        message = json.loads(text)
        assert message["y"][256] is None
        assert message["y"][0] == -1 and message["y"][-1] == 1
        assert len(message["x"]) == len(message["y"]) == 513


@pytest.mark.parametrize(
    "message",
    [
        "not json",
        "{}",
        json.dumps(viewport(xlim=[2, 1])),
        json.dumps(viewport(xlim=[0, 0])),
        json.dumps(viewport(xlim=[0, float("inf")])),
        json.dumps(viewport(xlim=[0, float("nan")])),
        json.dumps(viewport(xlim=[0])),
        json.dumps(viewport(xlim=[False, 1])),
        json.dumps(viewport(xlim=["0", "1"])),
        json.dumps(viewport(ylim=[2, 1])),
        json.dumps(viewport(-1)),
        json.dumps(viewport(2**53)),
        json.dumps(viewport(True)),
        json.dumps(viewport(extra=True)),
    ],
)
def test_invalid_messages_leave_connection_usable(viewer, message):
    with (
        TestClient(viewer._create_app()) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_text(message)
        assert websocket.receive_json()["type"] == "error"
        np.testing.assert_array_equal(viewer._xlim, [-5, 5])
        websocket.send_json(viewport())
        assert websocket.receive_json()["type"] == "samples"


def test_evaluation_errors_preserve_previous_state(viewer, monkeypatch):
    sample = viewer._sample

    def failing(bounds):
        if bounds[0] == 10:
            raise ValueError("Cannot evaluate this range")
        return sample(bounds)

    monkeypatch.setattr(viewer, "_sample", failing)
    with (
        TestClient(viewer._create_app()) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_json(viewport(7, xlim=[10, 20]))
        assert websocket.receive_json() == {
            "type": "error",
            "request_id": 7,
            "message": "Cannot evaluate this range",
        }
        np.testing.assert_array_equal(viewer._xlim, [-5, 5])
        websocket.send_json(viewport(8))
        assert websocket.receive_json()["request_id"] == 8


def test_browser_origin_must_match_local_server(viewer):
    with TestClient(viewer._create_app()) as client:
        with (
            pytest.raises(WebSocketDisconnect) as error,
            client.websocket_connect(
                "/ws", headers={"origin": "https://elsewhere.example"}
            ),
        ):
            pass
        assert error.value.code == 1008
        with client.websocket_connect(
            "/ws", headers={"origin": "http://testserver"}
        ) as websocket:
            assert websocket.receive_json()["type"] == "ready"


def test_background_server_reuses_closes_and_restarts(viewer, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    assert viewer.show(block=False) is viewer
    url = viewer.url
    assert url is not None and url.startswith("http://127.0.0.1:")
    assert opened == [url]
    thread = viewer._thread
    with urlopen(url, timeout=3) as response:
        assert b"Function graph" in response.read()
    with connect(url.replace("http://", "ws://") + "ws", open_timeout=3) as websocket:
        assert json.loads(websocket.recv(timeout=3))["type"] == "ready"
        websocket.send(json.dumps(viewport(9)))
        assert json.loads(websocket.recv(timeout=3))["request_id"] == 9
    viewer.show(block=False, open_browser=False)
    assert viewer.url == url and viewer._thread is thread
    assert opened == [url]
    viewer.close()
    viewer.close()
    assert viewer.url is None and thread is not None and not thread.is_alive()
    viewer.show(block=False, open_browser=False)
    assert viewer.url is not None and viewer._thread is not thread


def test_show_blocks_until_explicit_close(viewer):
    errors = []

    def run():
        try:
            viewer.show(open_browser=False)
        except BaseException as exc:
            errors.append(exc)

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    deadline = time.monotonic() + 10
    try:
        while viewer.url is None and caller.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert viewer.url is not None
        assert caller.is_alive()
    finally:
        viewer.close()
        caller.join(timeout=5)
    assert not caller.is_alive()
    assert not errors
    assert viewer.url is None


def test_startup_failure_is_clean_and_actionable(viewer, monkeypatch):
    def fail(self, sockets):
        raise RuntimeError("startup failure")

    monkeypatch.setattr(uvicorn.Server, "run", fail)
    with pytest.raises(RuntimeError, match="failed to start") as error:
        viewer.show(block=False, open_browser=False)
    assert error.value.__cause__ is not None
    assert viewer.url is None and viewer._socket is None and viewer._thread is None


def test_browser_open_failure_keeps_server_usable(viewer, monkeypatch):
    monkeypatch.setattr("webbrowser.open", lambda url: False)
    with pytest.warns(RuntimeWarning, match="manually"):
        viewer.show(block=False)
    assert viewer.url is not None


def test_keyboard_interrupt_stops_the_blocking_viewer(viewer, monkeypatch):
    import mathfirst.viz.viewer as runtime

    viewer.show(block=False, open_browser=False)
    thread = viewer._thread
    assert thread is not None
    sleep = runtime.time.sleep

    def interrupted(seconds):
        if threading.current_thread() is threading.main_thread():
            raise KeyboardInterrupt
        return sleep(seconds)

    monkeypatch.setattr(runtime.time, "sleep", interrupted)
    assert viewer.show(open_browser=False) is viewer
    assert viewer.url is None and not thread.is_alive()


def test_binary_messages_are_validated_without_losing_connection(viewer):
    with (
        TestClient(viewer._create_app()) as client,
        client.websocket_connect("/ws") as websocket,
    ):
        websocket.receive_json()
        websocket.send_bytes(b"\xff")
        assert websocket.receive_json()["type"] == "error"
        websocket.send_bytes(json.dumps(viewport()).encode())
        assert websocket.receive_json()["type"] == "samples"
