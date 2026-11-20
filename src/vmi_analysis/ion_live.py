"""Receive and accumulate the Go4-compatible Timepix3 preview stream."""

from collections import deque
from dataclasses import dataclass
from threading import Event, Lock, Thread

import numpy as np

from .ion_acquisition import PREFIX
from .tpx3_analysis import PIXEL_CLK_NS, FINE_NS, TDC_FINE_NS, TOA_ROLLOVER_NS
from .tof_bins import histogram_bin_edges


MAX_MESSAGE_BYTES = 16 * 1024 * 1024
GMD_HISTORY = 2_000


def gmd_histogram(values, low, high, bin_size, label):
    """Histogram the recent finite readings using independent manual or automatic limits."""
    if any(limit is not None and not np.isfinite(limit) for limit in (low, high)):
        raise ValueError(f"Enter finite {label} limits.")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"Enter increasing {label} limits.")
    if bin_size is not None and (not np.isfinite(bin_size) or bin_size <= 0):
        raise ValueError(f"{label} bin size must be finite and greater than zero.")
    finite = values[np.isfinite(values)]
    auto_low, auto_high = np.histogram_bin_edges(finite, bins=1) if finite.size else (0, bin_size or 1)
    span = bin_size or 1
    if low is None:
        low = auto_low if high is None or (finite.size and auto_low < high) else high - span
    if high is None:
        high = auto_high if auto_high > low else low + span
    edges = histogram_bin_edges(low, high, 40, bin_size=bin_size, label=label)
    return np.histogram(finite, bins=edges)


@dataclass
class LiveFrame:
    timestamp_ns: int
    gmd1_na: float
    gmd2_na: float
    pixels: np.ndarray
    tof_ns: np.ndarray
    trigger_count: int


def decode_message(message):
    """Read the acquisition prefix and little-endian packet words from one message."""
    if not PREFIX.size <= len(message) <= MAX_MESSAGE_BYTES or (len(message) - PREFIX.size) % 8:
        raise ValueError("Invalid TPX3 live message size.")
    seconds, nanoseconds, gmd1, gmd2 = PREFIX.unpack_from(message)
    if not 0 <= nanoseconds < 10**9:
        raise ValueError("Invalid acquisition timestamp.")
    words = np.frombuffer(message, dtype="<u8", offset=PREFIX.size)
    pixels = words[(words >> 60) == 0xB]
    subpixel = (pixels >> 44) & 7
    x = ((pixels >> 52) & 0xFE) + (subpixel >> 2)
    y = ((pixels >> 45) & 0xFC) + (subpixel & 3)
    addresses = (y * 256 + x).astype(np.intp)
    triggers = words[(words >> 56) == 0x6F]
    tof = np.empty(0)
    if len(triggers) == 1:
        trigger = int(triggers[0])
        fine = ((((trigger >> 5) & 15) - 1) << 9) // 12
        tdc = ((trigger >> 12) & 0xFFFFFFFF) * 25 + ((trigger & 0xE00) | (fine & 511)) * TDC_FINE_NS
        ctoa = (((pixels >> 30) & 0x3FFF) << 4) | (~(pixels >> 16) & 15)
        toa = (pixels & 0xFFFF).astype(float) * PIXEL_CLK_NS + ctoa * FINE_NS
        # Both counters wrap. Compare within one pixel clock period, preserving early hits.
        tof = (toa - tdc + 1_000) % TOA_ROLLOVER_NS - 1_000
    return LiveFrame(seconds * 10**9 + nanoseconds, gmd1 * 1e9, gmd2 * 1e9, addresses, tof, len(triggers))


class LivePreview:
    """One independent subscriber with bounded plot history and periodic snapshots."""

    def __init__(self, endpoint, edges):
        self.endpoint = endpoint
        self.edges = edges
        self._lock = Lock()
        self._stop = Event()
        self._thread = Thread(target=self._receive, daemon=True, name="live-tpx3")
        self.error = None
        self.reset()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def reset(self):
        with self._lock:
            self.image = np.zeros((256, 256), dtype=np.int64)
            self.histogram = np.zeros(len(self.edges) - 1, dtype=np.int64)
            self.gmd = deque(maxlen=GMD_HISTORY)
            self.origin = None
            self.frames = self.trigger_issues = self.invalid = 0

    def add(self, frame):
        image = np.bincount(frame.pixels, minlength=256 * 256).reshape(256, 256)
        histogram, _ = np.histogram(frame.tof_ns, bins=self.edges)
        with self._lock:
            if self.origin is None:
                self.origin = frame.timestamp_ns
            self.gmd.append(((frame.timestamp_ns - self.origin) / 1e9, frame.gmd1_na, frame.gmd2_na))
            self.image += image
            self.histogram += histogram
            self.frames += 1
            self.trigger_issues += frame.trigger_count != 1

    def snapshot(self):
        with self._lock:
            gmd = np.asarray(self.gmd, dtype=float).reshape(-1, 3)
            return self.image.copy(), self.histogram.copy(), gmd, self.frames, self.trigger_issues, self.invalid, self.error

    def _receive(self):
        try:
            import zmq

            with zmq.Context() as context, context.socket(zmq.SUB) as socket:
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 200)
                socket.setsockopt(zmq.RCVHWM, 64)
                socket.setsockopt(zmq.MAXMSGSIZE, MAX_MESSAGE_BYTES)
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(self.endpoint)
                while not self._stop.is_set():
                    try:
                        message = socket.recv_multipart()
                    except zmq.Again:
                        continue
                    try:
                        if len(message) != 1:
                            raise ValueError("Expected one complete frame per message.")
                        self.add(decode_message(message[0]))
                    except ValueError:
                        with self._lock:
                            self.invalid += 1
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
