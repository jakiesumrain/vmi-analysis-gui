"""Saved GUI settings, with a one-time read-only migration from legacy TOML."""

from pathlib import Path

from nicegui import app

from centroiding.options import load_settings, options, section

CENTROID_DIR = Path(__file__).resolve().parents[1] / "centroiding"
LEGACY_CONFIG = CENTROID_DIR / "config.toml"
MIGRATED_KEY = "centroid.settings_migrated"


def storage_key(group, key):
    return f"centroid.{group}.{key}"


def centroid_settings(store=None, legacy_path=None):
    """Copy current settings; existing saved keys always win over migration."""
    store = app.storage.general if store is None else store
    legacy_path = LEGACY_CONFIG if legacy_path is None else legacy_path
    if not store.get(MIGRATED_KEY):
        legacy = load_settings(legacy_path if Path(legacy_path).exists() else None)
        for group, key, _ in options():
            name = storage_key(group, key)
            if name not in store:
                store[name] = section(legacy, group)[key]
        store[MIGRATED_KEY] = True
    result = load_settings()
    for group, key, default in options():
        value = store.get(storage_key(group, key), default)
        if group == "spectrum_plot" and value is None:
            value = default
        section(result, group)[key] = value
    return result


def output_dir(config):
    path = Path(config["vmi_picasso"]["output_dir"])
    return path if path.is_absolute() else CENTROID_DIR / path


def binning_inputs(store=None):
    """Migrate the former Binning config picker to explicit saved inputs once."""
    store = app.storage.general if store is None else store
    config = centroid_settings(store)
    if not store.get("binning.inputs_migrated"):
        old_path = store.get("binning.config")
        if old_path and Path(old_path).is_file():
            config = load_settings(old_path)
        vmi = config["vmi_picasso"]
        defaults = {"checkpoint": str(output_dir(config) / vmi["checkpoint_file"]),
                    "output_dir": str(output_dir(config)), "gmd_map": vmi["gmd_map_path"],
                    "resolution_factor": vmi["histogram"]["resolution_factor"],
                    "gmd_weight_min": vmi["gmd"]["gmd_weight_min"],
                    "gmd_weight_max": vmi["gmd"]["gmd_weight_max"]}
        for key, value in defaults.items():
            if "binning." + key not in store:
                store["binning." + key] = value
        store["binning.inputs_migrated"] = True
    return {key: store["binning." + key] for key in (
        "checkpoint", "output_dir", "gmd_map", "resolution_factor", "gmd_weight_min", "gmd_weight_max")}


def binning_cli_args(config):
    """Explicit shared inputs for a binning job from a centroiding snapshot."""
    vmi = config["vmi_picasso"]
    return ["--checkpoint-path", str(output_dir(config) / vmi["checkpoint_file"]),
            "--output-dir", str(output_dir(config)), "--gmd-map-path", vmi["gmd_map_path"],
            "--resolution-factor", str(int(vmi["histogram"]["resolution_factor"])),
            "--gmd-weight-min", str(vmi["gmd"]["gmd_weight_min"]),
            "--gmd-weight-max", str(vmi["gmd"]["gmd_weight_max"])]
