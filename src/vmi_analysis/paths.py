"""Keep writable experiment files separate from installed application code.

In a checkout the workspace remains the repository root. Installed copies use
the launch directory, or VMI_WORKSPACE_DIR when explicitly supplied. Legacy
centroiding/circularization working directories retain their old meanings.
"""

import os
from pathlib import Path

_checkout = Path(__file__).resolve().parents[2]
_is_checkout = (_checkout / "pyproject.toml").is_file() and (_checkout / "src" / "vmi_analysis").resolve() == Path(__file__).resolve().parent
_default = _checkout if _is_checkout else Path.cwd()
WORKSPACE_ROOT = Path(os.environ.get("VMI_WORKSPACE_DIR", _default)).expanduser().resolve()
# Jobs change cwd to a stage's data directory. Children must retain the same
# workspace, especially for a wheel install with no checkout to discover.
os.environ["VMI_WORKSPACE_DIR"] = str(WORKSPACE_ROOT)
CENTROID_DIR = WORKSPACE_ROOT / "centroiding"
CIRCULARIZATION_DIR = WORKSPACE_ROOT / "circularization"
GENERATED_DIR = WORKSPACE_ROOT / ".cache" / "vmi-analysis"
STORAGE_DIR = WORKSPACE_ROOT / ".nicegui"
_old_cpbasex = WORKSPACE_ROOT / "pbasex-matlab"
CPBASEX_DIR = Path(os.environ.get("CPBASEX_DIR", _old_cpbasex if _old_cpbasex.is_dir()
                                else WORKSPACE_ROOT / "vendor" / "pbasex-matlab"))


def validate_distinct_output(output, inputs, *, cwd=None):
    """Resolve an output path and reject aliases of the named input files."""
    if not output:
        raise ValueError("Choose an output image file.")
    base = Path.cwd() if cwd is None else Path(cwd)
    target = (base / output).resolve()
    if target.is_dir():
        raise ValueError("Choose an output image file.")
    for label, source in inputs:
        if not source:
            continue
        source = (base / source).resolve()
        same_file = target == source
        if not same_file:
            try:
                same_file = target.samefile(source)
            except FileNotFoundError:
                pass
        if same_file:
            raise ValueError(f"The output image points to the {label}. Choose a different output file.")
    return target


def prepare_workspace():
    """Only runtime directories are created; user data is never relocated."""
    for path in (CENTROID_DIR, CIRCULARIZATION_DIR, GENERATED_DIR):
        path.mkdir(parents=True, exist_ok=True)
