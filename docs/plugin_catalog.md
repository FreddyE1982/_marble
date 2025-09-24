# Plugin Catalogue & Telemetry Quickstart

Marble now produces a live catalogue of every automatically registered plugin
alongside runtime telemetry for their activations.  The catalogue is generated
while the plugin loader scans `marble/plugins` and captures the following data
per plugin:

- plugin type (neuron, synapse, wanderer, brain-train, self-attention,
  neuroplasticity, or building block),
- implementation module and class,
- a functional niche inferred from the module and docstring vocabulary,
- an ``architecture_role`` string summarising the plugin's documented intent in
  ``ARCHITECTURE.md`` when available,
- the deterministic plugin identifier assigned by the loader,
- public hooks exposed by the plugin instance.

Both the catalogue and the telemetry feed into the central reporter so future
Mixture-of-Experts routing can reason about expert coverage and cost without
inspecting Python modules manually.

## Accessing the catalogue

```python
from marble.plugin_telemetry import get_plugin_catalog

catalogue = get_plugin_catalog()

conv1d = catalogue["conv1d"]
print(conv1d["niche"], conv1d["module"], conv1d["hooks"])
```

The same information is mirrored in the reporter tree under
`plugins/metadata/catalog`, making it available to dashboards and external
orchestration layers.

## Inspecting runtime telemetry

```python
from marble.plugin_telemetry import get_plugin_usage

usage = get_plugin_usage()
for name, stats in usage.items():
    print(name, stats["calls"], stats["avg_latency_ms"])
```

Telemetry entries are updated after every plugin hook invocation.  Each record
tracks activation counts, per-hook average latency, and the most recent latency
measurement.  Reporter consumers can read the data at
`plugins/metrics/usage` to drive routing policies or alerting.

When the Mixture-of-Experts router plugin (`moe_router`) is enabled its routing
decisions are mirrored to `decision_controller/moe_router/scalars`.  The live
metrics include active expert counts, load-balance variance, and latency budget
pressure—inputs the decision controller consumes to adjust cadence and
constraint multipliers on the fly.

For unit tests the helpers `reset_plugin_catalog()` and `reset_plugin_usage()`
provide a clean slate between scenarios.
