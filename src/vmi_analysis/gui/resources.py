"""File and shared-resource reservations for submitted command snapshots."""

from dataclasses import dataclass, replace
import os
from pathlib import Path

from vmi_analysis.centroiding.plot_data import (ABEL_IMAGE, RAW_IMAGE, SPECTRUM_CSV, SUMMARY_DATA,
                                                SUMMARY_PNG, TUNING_DATA, TUNING_PNG)
from vmi_analysis.paths import CENTROID_DIR, CIRCULARIZATION_DIR, CPBASEX_DIR


def canonical_path(value, cwd=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(cwd or Path.cwd()) / path
    return Path(os.path.normcase(str(path.resolve())))


def overlaps(first, second):
    return first == second or first.is_relative_to(second) or second.is_relative_to(first)


@dataclass(frozen=True)
class Access:
    reads: frozenset[Path] = frozenset()
    writes: frozenset[Path] = frozenset()
    exclusive: frozenset[str] = frozenset()
    gmd_map_output: Path | None = None
    centroid_output_dir: Path | None = None

    def union(self, other):
        return Access(self.reads | other.reads, self.writes | other.writes,
                      self.exclusive | other.exclusive)

    def _map_beside_centroid(self, other, output, directory):
        if (output != self.gmd_map_output or directory != other.centroid_output_dir
                or output == directory or not output.is_relative_to(directory)):
            return False
        # A differently named file can still be a hard link to a Centroiding file.
        for path in (other.writes - {directory}) | other.reads:
            try:
                if output.samefile(path):
                    return False
            except FileNotFoundError:
                continue
            except OSError:
                return False
        return True

    def conflict(self, other, *, allow_map_beside_centroid=False):
        shared = self.exclusive & other.exclusive
        if shared:
            return sorted(shared)[0]
        for write in self.writes:
            for path in other.reads | other.writes:
                if overlaps(write, path):
                    if (allow_map_beside_centroid and path not in other.reads
                            and (self._map_beside_centroid(other, write, path)
                                 or other._map_beside_centroid(self, path, write))):
                        continue
                    return str(path)
        for write in other.writes:
            for path in self.reads:
                if overlaps(write, path):
                    return str(path)
        return None


def command_access(cmd, cwd=None):
    """Declare actual GUI command inputs/outputs without importing analysis scripts.

    Output directories are reserved as a whole for multi-file analysis stages.
    A pipeline reserves the union for its entire run, including future stages.
    """
    if not cmd or len(cmd) < 3 or cmd[1] != "-m":
        return Access()
    module = cmd[2]
    args = [str(arg) for arg in cmd[3:]]

    def value(flag, default=""):
        for index, arg in enumerate(args):
            if arg == flag:
                return args[index + 1]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1]
        return default

    reads, writes, exclusive = set(), set(), set()
    gmd_map_output = centroid_output_dir = None

    def read(path, base=cwd):
        if path:
            reads.add(canonical_path(path, base))

    def write(path):
        if path:
            writes.add(canonical_path(path, cwd))

    if module == "vmi_analysis.gmd.build_gmd_map":
        read(value("--h5-dir")); read(value("--gmd-path"))
        output = value("--output-path", "gmd_map_all.txt")
        if output:
            gmd_map_output = canonical_path(output, cwd)
            writes.add(gmd_map_output)
    elif module == "vmi_analysis.imaging.h5_play_frames_movie":
        read(value("--h5-dir")); read(value("--dark-path"))
        write(value("--dump-frames"))
        if "--no-save-movie" not in args:
            name = Path(value("--h5-dir")).name
            directory = Path(value("--movie-dir") or cwd or Path.cwd())
            write(directory / (name + value("--movie-ext", ".mp4")))
            write(directory / (name + ".gif"))  # ffmpeg fallback
    elif module == "vmi_analysis.imaging.h5_accumulate_fast_gmd":
        read(value("--h5-dir")); read(value("--dark-path"))
        if "--enable-gmd" in args:
            read(value("--gmd-map-path"))
        output = value("--output-path", "accumulated_result.txt")
        if output:
            write(output); write(Path(output).with_suffix(".png"))
        write(value("--live-path"))
    elif module == "vmi_analysis.centroiding.compute_dark":
        read(value("--dark-dir"))
        directory = Path(value("--output-dir", str(CENTROID_DIR)))
        method = value("--method", "median")
        write(directory / f"dark_{method}.npy")
        write(directory / f"dark_{method}_info.txt")
    elif module == "vmi_analysis.centroiding.process_andor_vmi_PICASSO_v5_h5_batch":
        read(value("--h5-dir"))
        if "--dark-subtraction" in args:
            read(value("--dark-file"), CENTROID_DIR)
        if "--flat-field" in args:
            read(value("--flat-field-file"))
        if "--gmd-weighting" in args:
            read(value("--gmd-map-path"))
        centroid_output_dir = canonical_path(value("--output-dir", "vmi_output"), cwd)
        write(centroid_output_dir)
        # An absolute checkpoint filename can sit outside the output directory.
        write(centroid_output_dir / value("--checkpoint-file", "vmi_results_raw.h5"))
        for filename in (RAW_IMAGE, ABEL_IMAGE, SPECTRUM_CSV, SUMMARY_DATA, SUMMARY_PNG, TUNING_DATA, TUNING_PNG):
            write(centroid_output_dir / filename)
    elif module == "vmi_analysis.centroiding.vmi_histogram_gmd_bins":
        read(value("--checkpoint-path")); read(value("--gmd-map-path"))
        write(value("--output-dir"))
    elif module == "vmi_analysis.circularization.circularization":
        read(value("--input", "image.txt")); read(value("--extracted", "extracted.txt"))
        read(value("--reassign-file"))
        write(value("--output", "image_circularized.txt"))
        write(value("--polar-preview"))
        if "--no-plots" not in args:
            write(value("--plot-dir", "trig_fit_plots"))
    elif module == "vmi_analysis.circularization.process_gmd_bins":
        read(value("--bin-dir", str(CENTROID_DIR / "vmi_analysis_output_xe_j_100_sig")))
        for filename in ("extracted-FOC100eV-20260710.txt", "reassignment-FOC100eV-20260710.txt"):
            read(CIRCULARIZATION_DIR / filename)
        backend = value("--inversion-backend", "pyabel")
        name = "gmd_bins_results" + value("--weight-suffix", "_w-gmd_norm")
        if backend == "pyabel":
            name += "_pyabel_" + value("--pyabel-method", "daun")
        write(value("--output-dir", str(CIRCULARIZATION_DIR / name)))
        if backend == "matlab":
            exclusive.add("MATLAB/CPBASEX")
            write(CPBASEX_DIR / "examples")
    return Access(frozenset(reads), frozenset(writes), frozenset(exclusive),
                  gmd_map_output=gmd_map_output, centroid_output_dir=centroid_output_dir)


def job_access(job):
    access = Access(frozenset(canonical_path(p, job.cwd) for p in job.reads),
                    frozenset(canonical_path(p, job.cwd) for p in job.writes),
                    frozenset([*job.exclusive, f"panel:{job.name}"]))
    if job.stages is None:
        command = command_access(job.cmd, job.cwd)
        combined = access.union(command)
        if not access.writes:
            return replace(combined, gmd_map_output=command.gmd_map_output,
                           centroid_output_dir=command.centroid_output_dir)
        return combined
    for stage in job.stages:
        access = access.union(command_access(stage.cmd, stage.cwd))
        access = access.union(Access(writes=frozenset(canonical_path(p, stage.cwd) for p in stage.outputs)))
    return access
