"""Neutral defaults and explicit CLI inputs shared by centroiding and its GUI."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG = {"vmi_picasso": {
    "h5_dir": "", "h5_pattern": "RAW-*.h5", "h5_dataset": "raw/data",
    "h5_meta": "raw/meta", "meta_column_gmd": 1, "gmd_map_path": "",
    "dark_data_dir": "", "dark_h5_pattern": "RAW-*.h5", "dark_file": "",
    "output_dir": "vmi_output", "checkpoint_file": "vmi_results_raw.h5",
    "box_size": 5, "min_net_gradient": 400.0, "camera_baseline": 0.0,
    "camera_sensitivity": 1.0, "camera_gain": 1.0, "camera_qe": 0.9,
    "mle_convergence": 0.1, "mle_max_iterations": 50, "mle_method": "sigma",
    "fitting_method": "mle", "tuning_frame_index": 0,
    "tuning_bins_photons": 40, "tuning_bins_net_gradient": 40, "tuning_bins_precision": 40,
    "processing": {"enable_parallel": True, "num_cores": 6,
                   "enable_dark_subtraction": False, "dark_method": "mean",
                   "dark_frame_count": 2000, "enable_flat_field": False,
                   "flat_field_file": "", "enable_circular_mask": False,
                   "mask_center_x": None, "mask_center_y": None, "mask_radius": None,
                   "tuning_only": False},
    "histogram": {"resolution_factor": 1, "enable_abel": True, "abel_rotation_deg": 0.0},
    "energy_calibration": {"alpha": None, "fit_xe_4d": False,
                           "fit_min_px2": None, "fit_max_px2": None},
    "spectrum_plot": {"x_min": -1.0, "x_max": -1.0, "y_min": -1.0, "y_max": -1.0},
    "gmd": {"enable_gmd_weighting": False, "gmd_weight_min": 0.0, "gmd_weight_max": 4.0},
}}


def section(config, group):
    return config["vmi_picasso"] if group == "vmi_picasso" else config["vmi_picasso"][group]


def options():
    for key, value in DEFAULT_CONFIG["vmi_picasso"].items():
        if isinstance(value, dict):
            for child, default in value.items():
                yield key, child, default
        else:
            yield "vmi_picasso", key, value


def load_settings(config_path=None):
    """Read only an explicitly supplied TOML file; CLI defaults need no file."""
    result = deepcopy(DEFAULT_CONFIG)
    if config_path:
        with Path(config_path).open("rb") as handle:
            legacy = tomllib.load(handle)["vmi_picasso"]
        for group, key, _ in options():
            source = legacy if group == "vmi_picasso" else legacy.get(group, {})
            if key in source:
                section(result, group)[key] = source[key]
    return result


def flag_name(group, key):
    name = key.removeprefix("enable_")
    if group == "spectrum_plot":
        name = "spectrum_" + key
    return name.replace("_", "-")


def validate_settings(config, require_input=False):
    vmi = config["vmi_picasso"]
    if require_input and not str(vmi["h5_dir"]).strip():
        raise ValueError("Provide an HDF5 directory (--h5-dir)")
    if not str(vmi["output_dir"]).strip() or not str(vmi["checkpoint_file"]).strip():
        raise ValueError("Output directory and checkpoint filename must not be empty")
    for group, key, default in options():
        value = section(config, group)[key]
        if value is None:
            if default is not None:
                raise ValueError(f"{key} must not be empty")
            continue
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if isinstance(default, int) and not isinstance(default, bool):
            if not isinstance(value, (int, float)) or value != int(value):
                raise ValueError(f"{key} must be an integer")
    if vmi["box_size"] < 3 or vmi["box_size"] % 2 != 1:
        raise ValueError("Box size must be an odd integer of at least 3")
    if vmi["histogram"]["resolution_factor"] < 1 or vmi["processing"]["num_cores"] < 1:
        raise ValueError("Histogram resolution and CPU cores must be positive integers")
    if not isinstance(vmi["histogram"]["abel_rotation_deg"], (int, float)):
        raise ValueError("Abel rotation must be a number in degrees")
    # Import lazily so basic argument parsing does not import numerical libraries.
    if __package__:
        from .spectrum_calibration import CalibrationSettings
    else:
        from spectrum_calibration import CalibrationSettings
    CalibrationSettings.from_config(vmi["energy_calibration"])


def parse_centroid_args(argv=None):
    parser = argparse.ArgumentParser(description="Picasso centroiding and VMI spectrum analysis")
    parser.add_argument("--config", help="Optional legacy TOML defaults; explicit arguments take precedence")
    choices = {"mle_method": ["sigma", "sigmaxy"], "fitting_method": ["mle", "lq"],
               "dark_method": ["mean", "median"]}
    for group, key, default in options():
        kwargs = {"dest": f"{group}__{key}", "default": argparse.SUPPRESS}
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = type(default) if default is not None else float
        if key in choices:
            kwargs["choices"] = choices[key]
        if key == "abel_rotation_deg":
            kwargs["help"] = "Counterclockwise rotation after centering and before Abel inversion (default: 0 degrees)"
        parser.add_argument("--" + flag_name(group, key), **kwargs)
    args = parser.parse_args(argv)
    try:
        args.settings = load_settings(args.config)
        for group, key, _ in options():
            attr = f"{group}__{key}"
            if hasattr(args, attr):
                section(args.settings, group)[key] = getattr(args, attr)
        validate_settings(args.settings, require_input=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return args


def centroid_cli_args(config):
    """Serialize a settings snapshot, including explicit false boolean overrides."""
    validate_settings(config, require_input=True)
    result = []
    for group, key, default in options():
        value = section(config, group)[key]
        if value is None:
            continue
        flag = flag_name(group, key)
        if isinstance(default, bool):
            result.append("--" + ("" if value else "no-") + flag)
        else:
            if isinstance(default, int):
                value = int(value)
            result.append("--" + flag + "=" + str(value))
    return result
