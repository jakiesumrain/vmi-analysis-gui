"""File and shared-resource reservations for submitted command snapshots."""

from dataclasses import dataclass
import os
from pathlib import Path

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
    # Subset of ``writes`` that reserves a whole directory for a multi-file
    # stage. A distinct file written inside it by another job is allowed while
    # scheduling; reads and directory-vs-directory writes remain conservative.
    write_dirs: frozenset[Path] = frozenset()

    def __post_init__(self):
        # Keep the directory reservations visible in ``writes`` so every
        # read/write check sees them, even if a caller only sets write_dirs.
        if not self.write_dirs <= self.writes:
            object.__setattr__(self, "writes", self.writes | self.write_dirs)

    def union(self, other):
        return Access(self.reads | other.reads, self.writes | other.writes,
                      self.exclusive | other.exclusive,
                      self.write_dirs | other.write_dirs)

    def _file_writes(self):
        return self.writes - self.write_dirs

    def conflict(self, other):
        """Scheduling conflict.

        Reads conflict with any overlapping write. File writes conflict only on
        the same path. A file written inside another job's reserved output
        directory does not conflict, so a single-file stage can run alongside a
        multi-file stage that writes the same folder. Two reserved directories
        that overlap still conflict.
        """
        shared = self.exclusive & other.exclusive
        if shared:
            return sorted(shared)[0]
        for read in self.reads:
            for write in other.writes:
                if overlaps(read, write):
                    return str(write)
        for read in other.reads:
            for write in self.writes:
                if overlaps(read, write):
                    return str(write)
        for write in self._file_writes():
            for other_write in other.writes:
                if write == other_write:
                    return str(other_write)
        for write in other._file_writes():
            for other_write in self.writes:
                if write == other_write:
                    return str(write)
        for write in self.write_dirs:
            for other_dir in other.write_dirs:
                if overlaps(write, other_dir):
                    return str(other_dir)
        return None

    def conflict_strict(self, other):
        """Conservative conflict used for deletions and previews.

        Any overlapping write (including a path inside a reserved directory)
        conflicts, regardless of the scheduling relaxation above.
        """
        shared = self.exclusive & other.exclusive
        if shared:
            return sorted(shared)[0]
        for write in self.writes:
            for path in other.reads | other.writes:
                if overlaps(write, path):
                    return str(path)
        for write in other.writes:
            for read in self.reads:
                if overlaps(write, read):
                    return str(read)
        return None


def command_access(cmd, cwd=None):
    """Declare actual GUI command inputs/outputs without importing analysis scripts.

    Output directories are reserved as a whole for multi-file analysis stages:
    other jobs may not read inside them or reserve an overlapping directory,
    but they may write a distinct file inside them while scheduling.
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

    reads, writes, exclusive, write_dirs = set(), set(), set(), set()

    def read(path, base=cwd):
        if path:
            reads.add(canonical_path(path, base))

    def write(path):
        if path:
            writes.add(canonical_path(path, cwd))

    def write_dir(path):
        if path:
            directory = canonical_path(path, cwd)
            writes.add(directory)
            write_dirs.add(directory)

    if module == "vmi_analysis.gmd.build_gmd_map":
        read(value("--h5-dir")); read(value("--gmd-path"))
        write(value("--output-path", "gmd_map_all.txt"))
    elif module == "vmi_analysis.imaging.h5_play_frames_movie":
        read(value("--h5-dir")); read(value("--dark-path"))
        write_dir(value("--dump-frames"))
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
            dark = value("--dark-file")
            read(dark, CENTROID_DIR) if dark else read(value("--dark-data-dir"))
        if "--flat-field" in args:
            read(value("--flat-field-file"))
        if "--gmd-weighting" in args:
            read(value("--gmd-map-path"))
        output = canonical_path(value("--output-dir", "vmi_output"), cwd)
        write_dir(output)
        # An absolute checkpoint filename can sit outside the output directory.
        write(output / value("--checkpoint-file", "vmi_results_raw.h5"))
    elif module == "vmi_analysis.centroiding.vmi_histogram_gmd_bins":
        read(value("--checkpoint-path")); read(value("--gmd-map-path"))
        write_dir(value("--output-dir"))
    elif module == "vmi_analysis.circularization.circularization":
        read(value("--input", "image.txt")); read(value("--extracted", "extracted.txt"))
        read(value("--reassign-file"))
        write(value("--output", "image_circularized.txt"))
        if "--no-plots" not in args:
            write_dir(value("--plot-dir", "trig_fit_plots"))
    elif module == "vmi_analysis.circularization.process_gmd_bins":
        read(value("--bin-dir", str(CENTROID_DIR / "vmi_analysis_output_xe_j_100_sig")))
        for filename in ("extracted-FOC100eV-20260710.txt", "reassignment-FOC100eV-20260710.txt"):
            read(CIRCULARIZATION_DIR / filename)
        backend = value("--inversion-backend", "pyabel")
        name = "gmd_bins_results" + value("--weight-suffix", "_w-gmd_norm")
        if backend == "pyabel":
            name += "_pyabel_" + value("--pyabel-method", "daun")
        write_dir(value("--output-dir", str(CIRCULARIZATION_DIR / name)))
        if backend == "matlab":
            exclusive.add("MATLAB/CPBASEX")
            write_dir(CPBASEX_DIR / "examples")
    return Access(frozenset(reads), frozenset(writes), frozenset(exclusive),
                  frozenset(write_dirs))


def job_access(job):
    access = Access(frozenset(canonical_path(p, job.cwd) for p in job.reads),
                    frozenset(canonical_path(p, job.cwd) for p in job.writes),
                    frozenset([*job.exclusive, f"panel:{job.name}"]),
                    frozenset(canonical_path(p, job.cwd) for p in job.write_dirs))
    if job.stages is None:
        return access.union(command_access(job.cmd, job.cwd))
    for stage in job.stages:
        access = access.union(command_access(stage.cmd, stage.cwd))
        access = access.union(Access(writes=frozenset(canonical_path(p, stage.cwd) for p in stage.outputs)))
    return access
