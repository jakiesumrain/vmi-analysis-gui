"""Check the actual file interfaces of all GUI-backed analysis modules."""

from pathlib import Path
import sys
import tempfile
import unittest

from vmi_analysis.centroiding.options import centroid_cli_args, load_settings
from vmi_analysis.gui.jobs import Job, Stage
from vmi_analysis.gui.resources import Access, canonical_path, command_access, job_access
from vmi_analysis.paths import CENTROID_DIR, CIRCULARIZATION_DIR, CPBASEX_DIR


def access(module, args=(), cwd=None):
    return command_access([sys.executable, '-m', 'vmi_analysis.' + module, *args], cwd)


class JobResourcesTests(unittest.TestCase):
    def test_gmd_map_producer_conflicts_with_accumulate_reader_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as folder:
            producer = access('gmd.build_gmd_map', ['--h5-dir', 'raw', '--gmd-path', 'timestamps.txt',
                                                   '--output-path', 'map.txt'], folder)
            args = ['--h5-dir', 'raw', '--output-path', 'sum.txt', '--gmd-map-path', 'map.txt']
            reader = access('imaging.h5_accumulate_fast_gmd', args, folder)
            self.assertIsNone(producer.conflict(reader))
            reader = access('imaging.h5_accumulate_fast_gmd', [*args, '--enable-gmd'], folder)
            self.assertIsNotNone(producer.conflict(reader))
            self.assertIn(canonical_path('sum.png', folder), reader.writes)

    def test_dark_frame_producer_is_reserved_by_both_background_consumers(self):
        producer = access('centroiding.compute_dark', ['--dark-dir', 'raw-dark', '--method', 'mean',
                                                      '--output-dir', str(CENTROID_DIR)])
        settings = load_settings()
        vmi = settings['vmi_picasso']
        vmi.update(h5_dir='raw', dark_file='dark_mean.npy', gmd_map_path='map.txt')
        vmi['processing'].update(enable_dark_subtraction=True, enable_flat_field=True,
                                 flat_field_file='flat.npy')
        vmi['gmd']['enable_gmd_weighting'] = True
        centroid = access('centroiding.process_andor_vmi_PICASSO_v5_h5_batch', centroid_cli_args(settings), CENTROID_DIR)
        self.assertIsNotNone(producer.conflict(centroid))
        self.assertIn(canonical_path('flat.npy', CENTROID_DIR), centroid.reads)
        self.assertIn(canonical_path('map.txt', CENTROID_DIR), centroid.reads)
        self.assertIn(canonical_path('vmi_output', CENTROID_DIR), centroid.writes)
        movie = access('imaging.h5_play_frames_movie', ['--h5-dir', 'raw', '--dark-path',
                       str(CENTROID_DIR / 'dark_mean.npy'), '--dump-frames', 'preview', '--no-save-movie'])
        self.assertIsNotNone(producer.conflict(movie))
        self.assertEqual(movie.writes, frozenset([canonical_path('preview')]))

    def test_movie_output_includes_the_gif_fallback(self):
        movie = access('imaging.h5_play_frames_movie', ['--h5-dir', 'data', '--movie-dir', 'movies',
                                                      '--movie-ext', '.mp4'])
        self.assertEqual(movie.writes, frozenset([canonical_path('movies/data.mp4'), canonical_path('movies/data.gif')]))

    def test_binning_waits_for_the_entire_centroid_output_directory(self):
        centroid = access('centroiding.process_andor_vmi_PICASSO_v5_h5_batch', ['--output-dir', 'out'])
        bins = access('centroiding.vmi_histogram_gmd_bins', ['--checkpoint-path', 'out/events.h5',
                                                          '--gmd-map-path', 'map.txt', '--output-dir', 'bins'])
        self.assertIsNotNone(centroid.conflict(bins))

    def test_circularization_and_pes_inputs_outputs_and_matlab_lock(self):
        single = access('circularization.circularization', ['--input', 'image.txt', '--output', 'circ.txt',
                         '--extracted', 'fit.txt', '--reassign-file', 'radial.txt', '--plot-dir', 'plots'])
        self.assertIn(canonical_path('fit.txt'), single.reads)
        self.assertIn(canonical_path('radial.txt'), single.reads)
        self.assertIn(canonical_path('plots'), single.writes)
        quiet = access('circularization.circularization', ['--no-plots'])
        self.assertNotIn(canonical_path('trig_fit_plots'), quiet.writes)
        pyabel = access('circularization.process_gmd_bins', ['--bin-dir', 'bins', '--output-dir', 'pes'])
        matlab = access('circularization.process_gmd_bins', ['--bin-dir', 'bins', '--output-dir', 'other',
                                                           '--inversion-backend', 'matlab'])
        self.assertIsNone(pyabel.conflict(matlab))
        self.assertIn('MATLAB/CPBASEX', matlab.exclusive)
        self.assertIn(canonical_path(CPBASEX_DIR / 'examples'), matlab.writes)
        self.assertIn(canonical_path(CIRCULARIZATION_DIR / 'extracted-FOC100eV-20260710.txt'), pyabel.reads)

    def test_pipeline_reserves_outputs_of_all_stages_before_it_starts(self):
        pipeline = Job(name='Pipeline', stages=[
            Stage('map', [sys.executable, '-m', 'vmi_analysis.gmd.build_gmd_map', '--output-path', 'map.txt']),
            Stage('bins', [sys.executable, '-m', 'vmi_analysis.centroiding.vmi_histogram_gmd_bins',
                           '--checkpoint-path', 'events.h5', '--output-dir', 'bins'])])
        resources = job_access(pipeline)
        self.assertIn(canonical_path('map.txt'), resources.writes)
        self.assertIn(canonical_path('bins'), resources.writes)
        self.assertIn(canonical_path('events.h5'), resources.reads)
        self.assertIsNotNone(resources.conflict(Access(reads=frozenset([canonical_path('bins/bin1.npy')]))))


if __name__ == '__main__':
    unittest.main()
