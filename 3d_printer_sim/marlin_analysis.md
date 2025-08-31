# Marlin Firmware Analysis

## Hardware Abstraction Layers
Marlin organizes hardware support under `src/HAL/`, providing separate directories for each microcontroller family such as `AVR`, `STM32`, `ESP32`, and `TEENSY`. This structure cleanly isolates low-level implementations behind a common `HAL.h` interface.

## Required Interfaces
- **USB:** Virtual serial and mass-storage interfaces are implemented in `sd/usb_flashdrive` and related USB host headers.
- **SD Card:** File operations use the `sd` module, particularly `cardreader.cpp` and `SdVolume.h`.
- **Sensors:** Temperature and other sensors are handled in `module` code, e.g., temperature monitoring routines.
- **Motion Control:** Stepper motor management and axis movement live in `module` sources and rely on HAL pin mappings.

## Relevant Configuration Options
`Configuration.h` exposes key parameters needed for simulation:
- Build volume via `X_BED_SIZE`, `Y_BED_SIZE`, and `Z_MAX_POS`.
- Maximum print area derived from bed sizes and axis limits.
- Extruder count set by `EXTRUDERS` and related advanced options.
These settings must be mirrored in the simulator's configuration system.

## Configuration Enumeration Tool

To ensure comprehensive coverage, ``marlin_config_parser.py`` downloads
Marlin's ``Configuration.h`` and ``Configuration_adv.h`` and extracts all
``#define`` identifiers. This script keeps the simulator's development
plan aligned with upstream firmware options.
