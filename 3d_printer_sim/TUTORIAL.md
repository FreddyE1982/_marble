# 3D Printer Simulator Quick Start

1. Load configuration:

```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('config', pathlib.Path('3d_printer_sim/config.py'))
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
cfg = config.load_config('3d_printer_sim/config.yaml')
```

2. Create a microcontroller and thermal components:

```python
spec_mc = importlib.util.spec_from_file_location('mc', pathlib.Path('3d_printer_sim/microcontroller.py'))
microcontroller = importlib.util.module_from_spec(spec_mc)
spec_mc.loader.exec_module(microcontroller)
spec_s = importlib.util.spec_from_file_location('sensors', pathlib.Path('3d_printer_sim/sensors.py'))
sensors = importlib.util.module_from_spec(spec_s)
spec_s.loader.exec_module(sensors)
spec_t = importlib.util.spec_from_file_location('thermal', pathlib.Path('3d_printer_sim/thermal.py'))
thermal = importlib.util.module_from_spec(spec_t)
spec_t.loader.exec_module(thermal)
mc = microcontroller.Microcontroller()
therm = sensors.TemperatureSensor(mc, pin=1, value=25.0)
hotend = thermal.Hotend(
    therm,
    heating_rate=0.5,
    cooling_rate=0.25,
    temp_range=cfg.filament_types['PLA'].hotend_temp_range,
)
hotend.set_target_temperature(cfg.heater_targets['hotend'])
```

3. Step the simulation:

```python
hotend.update(1.0)  # advance by one second
print(therm.read_temperature())
```

This demonstrates configuration loading, filament temperature ranges, and target temperature control.

4. Run a basic motion and extrusion simulation with visualization:

```python
spec_sim = importlib.util.spec_from_file_location('sim', pathlib.Path('3d_printer_sim/simulation.py'))
sim = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(sim)
printer = sim.PrinterSimulation()
printer.set_axis_velocities(1, 0, 0)
printer.set_extrusion_velocity(100)
printer.update(1.0)  # move and extrude for one second
renderer = printer.visualizer.get_isometric_view()
# display(renderer)  # in a Jupyter notebook
```

The extruder moves along the X axis and deposits a short filament segment while the renderer stays in sync.

## Configuring Your Printer

To set up a printer model, create a `config.yaml` file with sections for
the build volume, motion limits, and extruders. A minimal example:

```yaml
build_volume: {x: 220, y: 220, z: 250}
bed_size: {x: 220, y: 220}
max_print_dimensions: {x: 200, y: 200, z: 200}
bed_tilt: {x: 0, y: 0}
bed_screws:
  front_left: 0
  front_right: 0
  back_left: 0
  back_right: 0
extruders:
  - steps_per_mm: 100
    filament_diameter: 1.75
filament_types:
  PLA:
    hotend_temp_range: [190, 220]
    bed_temp_range: [0, 60]
heater_targets:
  hotend: 200
  bed: 50
```

Load the configuration using the snippet at the top of this tutorial.
The loader validates numeric ranges and ensures required sections are
present. Adjust values to match your printer's specifications.

## Extending Hardware Components

Hardware modules follow simple patterns so they can be extended without
external dependencies. For example, a new sensor can be implemented as:

```python
from dataclasses import dataclass
from microcontroller import Microcontroller

@dataclass
class DummySensor:
    mc: Microcontroller
    pin: int
    value: float = 0.0

    def read_temperature(self) -> float:
        return self.value
```

After creating the sensor, register it with the microcontroller:

```python
mc = Microcontroller()
sensor = DummySensor(mc, pin=2)
mc.map_pin(2, sensor)
```

Any component that provides an `update(dt)` method can participate in the
simulation loop, enabling custom hardware to interact with the rest of
the system.
