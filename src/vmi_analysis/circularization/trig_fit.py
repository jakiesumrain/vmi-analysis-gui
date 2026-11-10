"""Batch clean-room extraction of trig-fit rings from a VMI image.

The interactive GUI tab is the usual workflow (manual band selection and
masking); this CLI repeats the same chain for explicit radial bands so the
result can be reproduced from a shell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vmi_analysis.imaging.centering import brightest_pixel

from . import polar_fit as pf


def build_parser():
    parser = argparse.ArgumentParser(
        description="Fit distorted VMI rings in a polar image and append "
                    "Gascooke-compatible trig-fit blocks.")
    parser.add_argument("--image", required=True,
                        help="2D NumPy detector image (.npy)")
    parser.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"),
                        help="image center in pixels (default: brightest pixel)")
    radial = parser.add_mutually_exclusive_group()
    radial.add_argument("--n-r", type=int,
                        help="number of radial samples (default 512)")
    radial.add_argument("--radial-bin-size", type=float,
                        help="spacing between radial samples in detector pixels")
    parser.add_argument("--n-theta", type=int, default=720,
                        help="number of angular bins (default 720)")
    parser.add_argument("--r-max", type=float, default=None,
                        help="maximum radius (default: largest circle inside)")
    parser.add_argument("--no-abel", action="store_true",
                        help="skip the inverse Abel transform")
    parser.add_argument("--abel-method", default="hansenlaw",
                        choices=["hansenlaw", "dasch_three_point",
                                 "dasch_two_point", "onion_peeling"])
    parser.add_argument("--band", nargs=2, type=float, action="append",
                        required=True, metavar=("R0", "R1"),
                        help="radial band for one ring; repeat for more rings")
    parser.add_argument("--mask", nargs=4, type=float, action="append",
                        default=[], metavar=("R0", "R1", "T0", "T1"),
                        help="mask rectangle (theta in degrees); repeatable")
    parser.add_argument("--theta0", type=float, default=0.0,
                        help="band start angle in degrees (default 0)")
    parser.add_argument("--theta1", type=float, default=360.0,
                        help="band end angle in degrees (default 360)")
    parser.add_argument("--n-terms", type=int, default=3)
    parser.add_argument("--no-beta", action="store_true",
                        help="do not fit the P2(cos theta) anisotropy term")
    parser.add_argument("--one-shot", action="store_true",
                        help="fit all trig terms at once instead of iteratively")
    parser.add_argument("--output", required=True,
                        help="extracted trig-fit file to append blocks to")
    parser.add_argument("--provenance", dest="provenance", action="store_true",
                        default=True)
    parser.add_argument("--no-provenance", dest="provenance",
                        action="store_false")
    parser.add_argument("--session", default=None,
                        help="optional session JSON output")
    parser.add_argument("--reduction", default="poly",
                        choices=["poly", "interp"],
                        help="model used for the printed a_n,b_n preview")
    parser.add_argument("--order", type=int, default=1,
                        help="polynomial order for the a_n,b_n preview")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if Path(args.image).suffix.lower() != ".npy":
        parser.error("--image must be a NumPy detector image (.npy)")
    if args.session:
        try:
            pf.session_output_path(args.session, image=args.image, calibration=args.output)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    image = pf.load_image(args.image)
    auto_center = args.center is None
    if auto_center:
        row, col = brightest_pixel(image)
        args.center = (col, row)
    n_r = 512 if args.n_r is None else args.n_r
    polar, theta, r, dr, _valid = pf.remap_polar(
        image, args.center, n_r=n_r, n_theta=args.n_theta,
        r_max=args.r_max, radial_bin_size=args.radial_bin_size)
    fit_image, _, options = pf.process_polar(
        polar, dr, abel=not args.no_abel, abel_method=args.abel_method)
    t0, t1 = np.deg2rad([args.theta0, args.theta1])
    masks = [pf.Rect(band[0], band[1], np.deg2rad(band[2]), np.deg2rad(band[3]))
             for band in args.mask]
    fits = []
    failures = 0
    for r0, r1 in args.band:
        try:
            fit = pf.fit_ring(fit_image, theta, r, pf.Rect(r0, r1, t0, t1),
                              masks=masks, n_terms=args.n_terms,
                              use_beta=not args.no_beta,
                              iterative=not args.one_shot,
                              processing=options)
            fit.detector_geometry = {"center": list(args.center), "shape": list(image.shape)}
            block = pf.ring_block_text(fit, provenance=args.provenance)
        except ValueError as exc:
            failures += 1
            print(f"band {r0}-{r1}: {exc}")
            continue
        fits.append(fit)
        pf.append_ring_block(args.output, block)
        print(f"ring r0={fit.params['r0']:.4f} sigma={fit.params['sigma']:.4f} "
              f"rms={fit.rms:.4g} converged={fit.converged} -> {args.output}")
    if args.session:
        session = {
            "image": {"path": str(args.image), "center": list(args.center),
                      "brightest_center": auto_center,
                      "n_theta": args.n_theta, "r_max": args.r_max},
            "processing": options,
            "masks": [[m.r0, m.r1, m.theta0, m.theta1] for m in masks],
            "rings": [fit.to_dict() for fit in fits],
            "output": str(args.output),
        }
        if args.radial_bin_size is None:
            session["image"]["n_r"] = n_r
        else:
            session["image"]["radial_bin_size"] = dr
        pf.save_session(args.session, session)
        print(f"session -> {args.session}")
    if len(fits) >= 2:
        try:
            reduced = pf.reduce_rings(fits, model=args.reduction,
                                      order=args.order)
        except ValueError as exc:
            print(f"reduction skipped: {exc}")
        else:
            for term in reduced["fit_meta"].values():
                if term.get("warning"):
                    print(f"Warning: {term['warning']}")
            print("coefficient preview:")
            for key, value in sorted(reduced["a"].items()):
                if not key.endswith("_sigma"):
                    print(f"  {key}: {value:.6g}")
    if failures:
        print(f"{failures} band(s) failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
