# VMI Analysis GUI

Analyze velocity-map imaging data from HDF5 camera frames. Use the panels
to locate electron spots, fit ring distortions, correct images, and calculate
energy spectra.

**Panel status.** Binning, PES, and Pipeline are immature and need restructuring.

## Get started

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
Run these commands from the checkout.

```sh
uv sync
uv run vmi-analysis
```

The app opens at [localhost:13517](http://localhost:13517). Set `VMI_GUI_PORT`
to use a different port. Supply your own camera files. For GMD matching,
also supply the GMD timestamp file.

Use **Save workflow** in **Settings** to save panel values to a JSON file.
Use **Load workflow** to restore those values. Save a Trig Calibration
session separately to preserve its band, masks, and ring fits.

## Documentation

- [User manual (PDF)](docs/manual.pdf) for installation, data formats, panel
  settings, the full pipeline, CLI commands, and troubleshooting.
- [Architecture](docs/architecture.md) and [contributing guide](CONTRIBUTING.md)
  for development.
- [Windows installer](packaging/windows/README.md) for offline installation
  and release builds.

## License

The project uses the [Apache License, Version 2.0](LICENSE). Bundled CPBASEX
uses its own [MIT license](vendor/pbasex-matlab/LICENSE.txt). See [NOTICE](NOTICE)
for attribution.
