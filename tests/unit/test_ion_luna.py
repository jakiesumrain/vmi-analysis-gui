from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import h5py

from vmi_analysis.gui.resources import canonical_path, command_access
from vmi_analysis.ion_luna import MAX_CHUNK_BYTES, display_input_path, main, luna_command


class LunaTests(unittest.TestCase):
    def test_native_command_and_windows_paths(self):
        args = ["--disable-clustering", "--eps-t", "125ns"]
        self.assertEqual(luna_command("source directory", "result file.h5", "matrix.h5", args, windows=False),
                         ["tpx3dump", "process", "-i", "source directory", "-o", "result file.h5",
                          *args, "--timewalk-file", "matrix.h5"])
        command = luna_command("E:\\source directory", "E:\\result file.h5", None, args,
                               "Ubuntu-20.04", "/opt/luna/bin/tpx3dump", windows=True)
        self.assertEqual(command[:7], ["wsl.exe", "--distribution", "Ubuntu-20.04", "--cd", "/", "--exec", "bash"])
        self.assertEqual(command[-8:], ["/opt/luna/bin/tpx3dump", "process", "E:\\source directory", "E:\\result file.h5", "", *args])
        self.assertEqual(luna_command("stats.h5", "corrected.h5", None, ["--ctot-cut", "500"],
                                     windows=False, correction=True),
                         ["tpx3dump", "apply-timewalk-correction", "--timewalk-file", "stats.h5",
                          "--output-file", "corrected.h5", "--ctot-cut", "500"])

    def test_reservations_cover_the_folder_matrix_and_output(self):
        access = command_access(["python", "-m", "vmi_analysis.ion_luna", "--input", "run",
                                 "--output", "final.h5", "--timewalk-file", "timewalk.h5",
                                 "--metadata-output", "metadata.h5"])
        self.assertEqual(access.reads, {canonical_path("run"), canonical_path("timewalk.h5")})
        self.assertEqual(access.writes, {canonical_path("final.h5"), canonical_path("metadata.h5")})

    def test_acquisition_metadata_and_packet_chunks(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "recording"
            source.mkdir()
            prefix = struct.pack("<qqdd", 1_794_485_586, 444_736_228, 1.5e-9, -2.5e-9)
            earlier = struct.pack("<qqdd", 1_794_485_586, 444_736_227, 3e-9, 4e-9)
            packet = b"\x00" * 7 + b"\xb0"
            long_body = packet * (MAX_CHUNK_BYTES // 8) + packet
            (source / "a_frame_12.tpx3").write_bytes(earlier + long_body)
            (source / "b_frame_13.tpx3").write_bytes(prefix)
            (source / "z_frame_2.tpx3").write_bytes(prefix + packet)
            output, metadata = root / "events.h5", root / "metadata.h5"

            def write_result(command, *, windows, original_paths, log):
                framed = Path(command[command.index("-i") + 1])
                staged = sorted(framed.iterdir())
                expected = (struct.pack("<4sBBH", b"TPX3", 0, 0, MAX_CHUNK_BYTES)
                            + long_body[:-8] + b"TPX3\x00\x00\x08\x00" + packet)
                self.assertEqual([p.read_bytes() for p in staged], [b"TPX3\x00\x00\x08\x00" + packet, expected])
                for staged_file, original in zip(staged, ("z_frame_2.tpx3", "a_frame_12.tpx3"), strict=True):
                    log_line = f"  - path: /mnt/e/temp/framed/{staged_file.name}\n"
                    self.assertEqual(display_input_path(log_line, original_paths), f"  - path: {source / original}\n")
                with h5py.File(metadata) as file:
                    frames = file["frames"]
                    self.assertEqual(frames.dtype.names, ("filename", "timestamp_ns", "gmd1", "gmd2"))
                    self.assertEqual(list(frames["filename"]), [b"z_frame_2.tpx3", b"a_frame_12.tpx3", b"b_frame_13.tpx3"])
                    self.assertEqual(tuple(frames[0]), (b"z_frame_2.tpx3", 1_794_485_586_444_736_228, 1.5e-9, -2.5e-9))
                    self.assertEqual(tuple(frames[1]), (b"a_frame_12.tpx3", 1_794_485_586_444_736_227, 3e-9, 4e-9))
                with h5py.File(command[command.index("-o") + 1], "w"):
                    pass

            arguments = ["--input", str(source), "--output", str(output), "--metadata-output", str(metadata)]
            with patch("vmi_analysis.ion_luna.sys.platform", "linux"), \
                    patch("vmi_analysis.ion_luna._run_luna", side_effect=write_result):
                main(arguments)
            self.assertEqual((source / "a_frame_12.tpx3").read_bytes(), earlier + long_body)
            self.assertTrue(output.is_file())

            (source / "b_frame_13.tpx3").write_bytes(prefix + b"bad")
            previous_metadata = metadata.read_bytes()
            with patch("vmi_analysis.ion_luna._run_luna") as run:
                with self.assertRaisesRegex(SystemExit, "incomplete 8-byte Timepix3 packet"):
                    main(arguments)
                run.assert_not_called()
            self.assertEqual(metadata.read_bytes(), previous_metadata)

    def test_correction_uses_separate_output(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "stats.h5"
            output = Path(folder) / "corrected.h5"
            with h5py.File(source, "w") as h5:
                for name in ("TimewalkMatrix", "PixelHits", "Clusters"):
                    h5.create_dataset(name, data=[1])
            original = source.read_bytes()

            def correct(command, *, windows, original_paths, log):
                self.assertEqual(command[1:4], ["apply-timewalk-correction", "--timewalk-file", str(source)])
                pending = Path(command[command.index("--output-file") + 1])
                self.assertNotEqual(pending, output)
                with h5py.File(pending, "w") as h5:
                    h5.create_dataset("Clusters", data=[2])

            args = ["--apply-timewalk-correction", "--input", str(source), "--output", str(output)]
            with patch("vmi_analysis.ion_luna.sys.platform", "linux"), \
                    patch("vmi_analysis.ion_luna._run_luna", side_effect=correct):
                main(args)
            self.assertEqual(source.read_bytes(), original)
            with h5py.File(output) as file:
                self.assertEqual(list(file["Clusters"]), [2])

    def test_prefixed_timewalk_calibration_requires_one_stream_and_valid_statistics(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "recording"
            source.mkdir()
            output, metadata = root / "master.h5", root / "master_metadata.h5"

            def write_frame(index, seconds, coarse):
                trigger = (0x6F << 56) | (coarse << 12) | (1 << 5)
                (source / f"frame_{index}.tpx3").write_bytes(struct.pack("<qqddQ", seconds, 0, 1, 2, trigger))

            args = ["--input", str(source), "--output", str(output), "--metadata-output", str(metadata),
                    "--calculate-timewalk-statistics", "--timewalk-calibration"]
            write_frame(1, 1, 100)
            write_frame(2, 2, 50)
            with patch("vmi_analysis.ion_luna._run_luna") as run:
                with self.assertRaisesRegex(SystemExit, "acquisition restart"):
                    main(args)
                run.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(metadata.exists())

            write_frame(2, 2, 200)
            with patch("vmi_analysis.ion_luna.sys.platform", "linux"), \
                    patch("vmi_analysis.ion_luna._run_luna") as run:
                def write_result(command, **_kwargs):
                    framed = Path(command[command.index("-i") + 1])
                    self.assertEqual(framed.name, "acquisition.tpx3")
                    self.assertEqual(framed.read_bytes().count(b"TPX3"), 2)
                    with h5py.File(command[command.index("-o") + 1], "w") as h5:
                        h5.create_dataset("TimewalkMatrix", data=[1])
                        if run.call_count == 2:
                            h5.create_dataset("TimewalkLookupTable", data=[2])

                run.side_effect = write_result
                with self.assertRaisesRegex(SystemExit, "nonempty /TimewalkLookupTable"):
                    main(args)
                self.assertFalse(output.exists())
                self.assertFalse(metadata.exists())
                main(args)
            self.assertTrue(metadata.is_file())
            with h5py.File(output) as h5:
                self.assertEqual(h5["TimewalkLookupTable"][0], 2)

    def test_validation_and_failed_conversion_preserve_existing_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "frame.tpx3"
            source.write_bytes(b"incompatible")
            target = root / "result.h5"
            target.write_bytes(b"previous result")
            args = ["--input", str(source), "--output", str(target)]
            with patch("vmi_analysis.ion_luna._run_luna") as run:
                with self.assertRaisesRegex(SystemExit, "Serval TPX3 chunk header"):
                    main(args)
                run.assert_not_called()
            source.write_bytes(b"TPX3\0\0\0\0")
            with patch("vmi_analysis.ion_luna._run_luna", side_effect=ValueError("test conversion failed")):
                with self.assertRaisesRegex(SystemExit, "test conversion failed"):
                    main(args)
            self.assertEqual(target.read_bytes(), b"previous result")

    def test_folder_outputs_are_published_after_conversion(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "recording"
            source.mkdir()
            (source / "a.tpx3").write_bytes(b"TPX3\0\0\0\0")
            (source / "b.tpx3").write_bytes(b"TPX3\0\0\0\0")
            output = root / "events.h5"

            def write_result(command, *, windows, original_paths, log):
                with h5py.File(command[command.index("-o") + 1], "w") as file:
                    file.create_dataset("Clusters", data=[1, 2])

            with patch("vmi_analysis.ion_luna.sys.platform", "linux"), \
                    patch("vmi_analysis.ion_luna._run_luna", side_effect=write_result):
                main(["--input", str(source), "--output", str(output)])
            with h5py.File(output) as file:
                self.assertEqual(list(file["Clusters"]), [1, 2])
