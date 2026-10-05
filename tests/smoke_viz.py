"""Exercise the installed viewer, including bundled resources and sockets."""

import json
from urllib.request import urlopen

from websockets.sync.client import connect

from mathfirst import Polynomial, Variable
from mathfirst.viz import FunctionGraph, Viewer

x = Variable("x")
viewer = Viewer(FunctionGraph(Polynomial(x**3 - 2 * x + 1, x)))
try:
    viewer.show(block=False, open_browser=False)
    url = viewer.url
    assert url is not None
    for path in ("", "assets/viewer.js", "assets/viewer.css", "assets/plotly.min.js"):
        with urlopen(url + path, timeout=5) as response:
            assert response.status == 200 and response.read()
    with connect(url.replace("http://", "ws://") + "ws", open_timeout=5) as websocket:
        assert json.loads(websocket.recv(timeout=5))["type"] == "ready"
        websocket.send(
            json.dumps({"type": "viewport", "request_id": 1, "xlim": [-1, 1]})
        )
        message = json.loads(websocket.recv(timeout=5))
        assert message["type"] == "samples"
        assert message["y"][0] == 2 and message["y"][-1] == 0
finally:
    viewer.close()
assert viewer.url is None
print("Installed visualization extra and bundled assets work")
