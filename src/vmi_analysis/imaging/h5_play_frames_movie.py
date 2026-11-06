#!/usr/bin/env python3
"""
Play the first N frames of the H5 raw data as a movie, and save it to a file.

Reads frames in acquisition order (h5 files sorted by name, frames in file
order) and displays them one by one in an animated matplotlib window, to
inspect the frame-to-frame pattern (VMI spot structure, background,
detector artifacts, etc.).

The movie is played first and written to disk afterwards, once the playback
window is closed.  Its filename is derived from the h5 data directory name
(e.g. /data/run_9 -> run_9.mp4) and it goes to `movie_dir`
(default: the current working directory).  MP4 writing needs ffmpeg on
PATH; if it is missing, the movie is written as a GIF instead.

The color scale is FIXED for the whole movie (estimated from the first few
frames) so that frame-to-frame intensity changes are directly comparable.
"""

import argparse
import sys
import h5py
import numpy as np
import inspect
import os
import time

import matplotlib
if "--headless" in sys.argv or os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.widgets import Button

# --- default config (override via CLI) ---
h5_dir      = ""                # required: directory containing the RAW-*.h5 files
h5_dataset  = "raw/data"      # frame stack dataset path
n_frames    = 20            # total number of frames to play (across files)
interval_ms = 500              # delay between frames in ms (50 -> 20 FPS)
cmap        = "gray"           # display colormap
log_scale   = False           # log color scale (recommended for sparse VMI data)
vmin_frac   = 0.5            # log-scale floor as fraction of vmax
vmax_pct    = 99.9           # vmax percentile (of the first frames' max image)
scale_frames = 10             # number of leading frames used to estimate vmax
dark_path   = r""             # optional dark frame .npy; empty string to skip
save_movie  = True            # also write the movie to a file
movie_dir   = r""             # output folder; empty -> current working directory
movie_ext   = ".mp4"          # ".mp4" (needs ffmpeg) or ".gif" (Pillow only)
movie_dpi   = 100             # resolution of the saved movie
movie_fps   = 0               # saved-movie frame rate; 0 -> same speed as playback
headless    = False           # skip the playback window (for web/GUI use)
dump_frames = r""             # optional dir for per-frame PNGs; empty to skip
stream_pace = False           # dump frames at playback pace (live web view)
# ------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Play/save the first N frames of H5 raw data as a movie.")
    p.add_argument("--h5-dir", default=h5_dir)
    p.add_argument("--h5-dataset", default=h5_dataset)
    p.add_argument("--n-frames", type=int, default=n_frames)
    p.add_argument("--interval-ms", type=int, default=interval_ms)
    p.add_argument("--cmap", default=cmap)
    p.add_argument("--log-scale", action="store_true", default=log_scale)
    p.add_argument("--vmin-frac", type=float, default=vmin_frac)
    p.add_argument("--vmax-pct", type=float, default=vmax_pct)
    p.add_argument("--scale-frames", type=int, default=scale_frames)
    p.add_argument("--dark-path", default=dark_path)
    p.add_argument("--save-movie", action=argparse.BooleanOptionalAction,
                   default=save_movie)
    p.add_argument("--movie-dir", default=movie_dir)
    p.add_argument("--movie-ext", default=movie_ext, choices=[".mp4", ".gif"])
    p.add_argument("--movie-dpi", type=int, default=movie_dpi)
    p.add_argument("--movie-fps", type=float, default=movie_fps)
    p.add_argument("--headless", action="store_true", default=headless,
                   help="Skip the playback window; only write outputs.")
    p.add_argument("--dump-frames", default=dump_frames,
                   help="Directory for per-frame PNGs (empty = skip).")
    p.add_argument("--stream-pace", action="store_true", default=stream_pace,
                   help="Dump frames at playback pace (interval-ms between "
                        "frames) so a web UI can follow the movie live.")
    args = p.parse_args()
    if not args.h5_dir:
        p.error("--h5-dir is required "
                "(no dataset is bundled with this repository).")
    return args


def saved_fps():
    """Frame rate for the saved movie (0 in the config -> match the playback)."""
    return float(movie_fps) if movie_fps > 0 else 1000.0 / max(1, interval_ms)


def movie_output_path():
    """Movie path whose file name is derived from the h5 data folder name."""
    name = os.path.basename(os.path.normpath(h5_dir))
    folder = movie_dir or os.getcwd()
    return os.path.join(folder, name + movie_ext)


def list_frames(h5_dir, n_frames):
    """Return [(file_idx, path, frame_idx, global_idx), ...] for the first n_frames.

    Files are read sorted by name, frames in file order.  Only the dataset
    shapes are inspected here, so memory usage stays small; the pixel data of
    one frame is loaded on demand by load_frame().
    """
    files = sorted(os.path.join(h5_dir, f)
                   for f in os.listdir(h5_dir) if f.endswith('.h5'))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    frames = []
    for file_idx, fpath in enumerate(files):
        if len(frames) >= n_frames:
            break
        with h5py.File(fpath, 'r') as hf:
            n_here = min(hf[h5_dataset].shape[0], n_frames - len(frames))
        for i in range(n_here):
            frames.append((file_idx, fpath, i, len(frames)))
    return frames


def load_frame(rec):
    """Read the pixel data of one frame listed by list_frames()."""
    _, fpath, frame_idx, _ = rec
    with h5py.File(fpath, 'r') as hf:
        return hf[h5_dataset][frame_idx]


def preprocess(frame, dark):
    """Dark-subtract and clip at 0 so the log color scale is well defined."""
    frame = frame.astype(np.float64)
    if dark is not None:
        frame -= dark
    return np.clip(frame, 0.0, None)


def save_animation(anim, out_path, n_frames):
    """Write the animation to out_path (.mp4 via ffmpeg, .gif via Pillow).

    Falls back to GIF when mp4 was requested but ffmpeg is not installed.
    """
    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)

    if out_path.lower().endswith('.mp4') and not FFMpegWriter.isAvailable():
        out_path = os.path.splitext(out_path)[0] + '.gif'
        print("ffmpeg not found: writing GIF instead of MP4")

    fps = saved_fps()
    if out_path.lower().endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps)

    step = max(1, n_frames // 20)

    def progress(i, n):
        if i % step == 0 or i == n - 1:
            done = 100.0 * (i + 1) / n
            print(f"  writing movie: frame {i + 1}/{n} ({done:5.1f}%)",
                  end='\n' if i == n - 1 else '\r', flush=True)

    print(f"Saving movie ({fps:g} fps) to {out_path} ...")
    save_kwargs = dict(writer=writer, dpi=movie_dpi)
    if 'progress_callback' in inspect.signature(anim.save).parameters:
        save_kwargs['progress_callback'] = progress   # matplotlib >= 3.4
    anim.save(out_path, **save_kwargs)
    print(f"Movie written: {out_path}")


def main():
    dark = np.load(dark_path) if dark_path and os.path.exists(dark_path) else None
    if dark is not None:
        print(f"Dark frame loaded: mean={dark.mean():.1f}, std={dark.std():.1f}")

    frames = list_frames(h5_dir, n_frames)
    if not frames:
        print("No frames found.")
        return
    n_total = len(frames)
    print(f"Playing {n_total} frames from {os.path.normpath(h5_dir)}")

    # estimate a fixed vmax from the first frames' running-max image, so
    # frame-to-frame intensity differences stay comparable in the movie
    head = [preprocess(load_frame(r), dark) for r in frames[:scale_frames]]
    vmax = float(np.percentile(np.max(np.stack(head), axis=0), vmax_pct))
    print(f"Display scale: vmax = {vmax:.1f} ({vmax_pct}th pct of first "
          f"{len(head)} frames), log floor = {vmax * vmin_frac:.4f}")

    norm = (LogNorm(vmin=vmax * vmin_frac, vmax=vmax) if log_scale
            else matplotlib.colors.Normalize(vmin=vmax * vmin_frac, vmax=vmax))

    # optional per-frame PNG dump (for the GUI scrub slider; with
    # --stream-pace the dump runs at playback speed for the live web view)
    if dump_frames:
        os.makedirs(dump_frames, exist_ok=True)
        for rec in frames:
            img = preprocess(load_frame(rec), dark)
            plt.imsave(os.path.join(dump_frames, f"frame_{rec[3]:04d}.png"),
                       norm(img), cmap=cmap, vmin=0.0, vmax=1.0)
            if stream_pace:
                print(f"[stream] frame {rec[3] + 1}/{n_total}", flush=True)
                time.sleep(max(0, interval_ms) / 1000.0)
        print(f"Dumped {n_total} frame PNGs to {dump_frames}")

    fig, ax = plt.subplots(figsize=(7, 7.8))
    fig.subplots_adjust(bottom=0.12)   # room for the pause/resume button
    fig.patch.set_facecolor('white')
    im = ax.imshow(head[0], cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label='counts' + (' (log)' if log_scale else ''))
    title = ax.set_title("", fontsize=11, fontweight='bold')

    # pause/resume button (hidden while the movie is being written)
    ax_btn = fig.add_axes([0.35, 0.02, 0.3, 0.045])
    btn = Button(ax_btn, 'Pause', hovercolor='0.85')
    state = {'paused': False, 'saving': False}

    def toggle_pause(_event):
        if state['paused']:
            anim.event_source.start()
            btn.label.set_text('Pause')
        else:
            anim.event_source.stop()
            btn.label.set_text('Resume')
        state['paused'] = not state['paused']

    btn.on_clicked(toggle_pause)

    def update(t):
        file_idx, _path, frame_idx, g = frames[t]
        im.set_data(preprocess(load_frame(frames[t]), dark))
        title.set_text(f"File {file_idx}, frame {frame_idx}   "
                       f"(global {g + 1}/{n_total})")
        if not state['saving'] and t == n_total - 1:
            print("\nEnd of movie.")
        return [im, title]

    # only build the animation if it will actually be rendered (playback
    # window or movie file) — otherwise matplotlib warns that it was
    # deleted without rendering anything
    anim = (FuncAnimation(fig, update, frames=range(n_total),
                          interval=interval_ms, blit=False,
                          cache_frame_data=False)
            if not headless or save_movie else None)

    # play the movie first (skipped in headless mode) ...
    if not headless:
        plt.show()

    # ... then write it to a file
    out_path = movie_output_path()
    if save_movie:
        state['saving'] = True
        ax_btn.set_visible(False)          # keep the button out of the video
        save_animation(anim, out_path, n_total)
        ax_btn.set_visible(True)
        state['saving'] = False
    else:
        print(f"Movie saving disabled (save_movie = False); "
              f"would have been written to {out_path}")


if __name__ == '__main__':
    _args = parse_args()
    h5_dir = _args.h5_dir
    h5_dataset = _args.h5_dataset
    n_frames = _args.n_frames
    interval_ms = _args.interval_ms
    cmap = _args.cmap
    log_scale = _args.log_scale
    vmin_frac = _args.vmin_frac
    vmax_pct = _args.vmax_pct
    scale_frames = _args.scale_frames
    dark_path = _args.dark_path
    save_movie = _args.save_movie
    movie_dir = _args.movie_dir
    movie_ext = _args.movie_ext
    movie_dpi = _args.movie_dpi
    movie_fps = _args.movie_fps
    headless = _args.headless
    dump_frames = _args.dump_frames
    stream_pace = _args.stream_pace
    main()
