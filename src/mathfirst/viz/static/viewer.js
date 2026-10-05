"use strict";

(() => {
  const graph = document.getElementById("graph");
  const status = document.getElementById("status");
  const reset = document.getElementById("reset");
  const expression = document.getElementById("expression");
  const socket = new WebSocket(`ws://${location.host}/ws`);
  let view = null;
  let initialRange = null;
  let variable = "x";
  let latestRequest = 0;
  let revision = 0;
  let updating = false;
  let timer = null;
  let messages = Promise.resolve();

  function setStatus(text, error = false) {
    status.textContent = text;
    status.dataset.error = String(error);
  }

  function layout() {
    return {
      margin: { l: 65, r: 25, t: 25, b: 55 },
      font: { family: "system-ui, sans-serif", color: "#243145" },
      dragmode: "pan",
      showlegend: false,
      uirevision: revision,
      xaxis: { title: { text: variable }, range: view.xlim, autorange: false, zerolinecolor: "#aeb8c8" },
      yaxis: { title: { text: "f(x)" }, ...(view.ylim ? { range: view.ylim, autorange: false } : { autorange: true }), zerolinecolor: "#aeb8c8" },
    };
  }

  const config = { responsive: true, scrollZoom: true, displayModeBar: false, doubleClick: false };

  function sendViewport() {
    if (!view || socket.readyState !== WebSocket.OPEN) return;
    socket.send(JSON.stringify({ type: "viewport", request_id: latestRequest, xlim: view.xlim, ylim: view.ylim }));
    setStatus("Updating…");
  }

  function requestSamples(delay = 0) {
    latestRequest += 1;
    clearTimeout(timer);
    timer = setTimeout(sendViewport, delay);
  }

  function changedRange(event, axis, previous) {
    if (event[`${axis}.range`]) return event[`${axis}.range`];
    if (event[`${axis}.range[0]`] !== undefined && event[`${axis}.range[1]`] !== undefined) {
      return [event[`${axis}.range[0]`], event[`${axis}.range[1]`]];
    }
    return previous;
  }

  async function handle(message) {
    if (message.type === "ready") {
      variable = message.variable;
      expression.textContent = message.label;
      initialRange = message.initial_xlim;
      view = { xlim: message.xlim, ylim: message.ylim };
      await Plotly.newPlot(graph, [], layout(), config);
      graph.on("plotly_relayout", event => {
        if (updating) return;
        const rangeEvent = Object.keys(event).some(key => /^[xy]axis\.(range|autorange)/.test(key));
        if (!rangeEvent) return;
        view.xlim = event["xaxis.autorange"] ? [...initialRange] : changedRange(event, "xaxis", view.xlim);
        view.ylim = event["yaxis.autorange"] ? null : changedRange(event, "yaxis", view.ylim);
        requestSamples(120);
      });
      reset.disabled = false;
      requestSamples();
    } else if (message.type === "samples") {
      if (message.request_id !== latestRequest) return;
      updating = true;
      try {
        await Plotly.react(graph, [{ x: message.x, y: message.y, type: "scatter", mode: "lines", connectgaps: false, line: { color: "#3868ad", width: 2 }, hovertemplate: "%{x}, %{y}<extra></extra>" }], layout(), config);
      } finally {
        updating = false;
      }
      setStatus(message.y.some(value => value !== null) ? "Connected" : "No finite real values in this range.");
    } else if (message.type === "error" && (message.request_id === null || message.request_id === latestRequest)) {
      setStatus(message.message, true);
    }
  }

  reset.addEventListener("click", () => {
    view = { xlim: [...initialRange], ylim: null };
    revision += 1;
    requestSamples();
  });
  socket.onmessage = event => {
    messages = messages.then(() => handle(JSON.parse(event.data))).catch(() => {
      setStatus("Could not update the graph.", true);
    });
  };
  socket.onerror = () => setStatus("Could not connect to the viewer.", true);
  socket.onclose = () => {
    clearTimeout(timer);
    reset.disabled = true;
    setStatus("Disconnected. Keep the Python viewer running to interact.", true);
  };
})();
