# VMI Analysis GUI

Analyze velocity-map imaging data from raw camera HDF5 frames through
centroiding, GMD binning, circularization, and Abel inversion. The web
interface runs individual analysis tools or the full pipeline.

## Get started

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/), then run
these commands from the checkout.

```sh
uv sync
uv run vmi-analysis
```

The app opens at [localhost:3157](http://localhost:3157). Set `VMI_GUI_PORT`
to use a different port. Bring your own `RAW-*.h5` files and a GMD timestamp
file for GMD matching. The manual describes the calibration files needed
for PES.

Use **Settings → Save workflow** to save panel values to a JSON file and
**Load workflow** to restore them before running an analysis.

## Documentation

- [User manual (PDF)](docs/manual.pdf) for installation, data formats, panel
  settings, the full pipeline, CLI commands, and troubleshooting.
- [Architecture](docs/architecture.md) and [contributing guide](CONTRIBUTING.md)
  for development.

## License

The project uses the [Apache License, Version 2.0](LICENSE). Bundled CPBASEX
uses its own [MIT license](vendor/pbasex-matlab/LICENSE.txt). See [NOTICE](NOTICE)
for attribution.
