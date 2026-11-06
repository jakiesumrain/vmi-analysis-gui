"""Real subprocess checks for scheduling, resource locks and independent stops."""

from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from vmi_analysis.gui.jobs import Job, JobManager, ResourceBusyError, Stage


def wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for concurrent job state")


class ConcurrentJobTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.manager = JobManager(max_concurrent=3)
        self.addCleanup(self.manager.shutdown)

    def job(self, name, **kwargs):
        gate = self.root / (name + '.release')
        code = "import sys,time; from pathlib import Path; print(sys.argv[2],flush=True)\nwhile not Path(sys.argv[1]).exists(): time.sleep(.01)"
        return Job(name=name, cmd=[sys.executable, '-c', code, str(gate), name], **kwargs)

    def release(self, job):
        (self.root / (job.name + '.release')).touch()
        wait_for(lambda: job.status == 'done')

    def ready(self, *jobs):
        wait_for(lambda: all(job._proc is not None and job.name in job.log for job in jobs))

    def test_three_independent_jobs_run_together_and_stop_separately(self):
        jobs = [self.manager.submit(self.job(str(i))) for i in range(4)]
        self.ready(*jobs[:3])
        self.assertEqual(jobs[3].status, 'queued')
        self.assertEqual(len({j._proc.pid for j in jobs[:3]}), 3)
        self.manager.cancel(jobs[0].id)
        self.ready(jobs[3])
        self.assertEqual(jobs[0].status, 'cancelled')
        self.assertTrue(all(j.status == 'running' for j in jobs[1:]))
        self.manager.shutdown()
        self.assertTrue(all(j.status == 'cancelled' for j in jobs))
        self.assertEqual(self.manager.running(), [])

    def test_limit_can_increase_above_two_and_decrease_without_stopping_jobs(self):
        self.manager.set_max_concurrent(2)
        jobs = [self.manager.submit(self.job(str(i))) for i in range(5)]
        self.ready(*jobs[:2])
        self.manager.set_max_concurrent(4)
        self.ready(*jobs[:4])
        self.manager.set_max_concurrent(1)
        self.assertTrue(all(j.status == 'running' for j in jobs[:4]))
        for job in jobs[:3]:
            self.release(job)
        self.assertEqual(jobs[4].status, 'queued')
        self.release(jobs[3])
        self.ready(jobs[4])
        for invalid in (0, -1, 2.5, True, float('inf'), None):
            with self.assertRaises(ValueError):
                self.manager.set_max_concurrent(invalid)

    def test_readers_share_files_writer_waits_and_unrelated_work_bypasses(self):
        self.manager.set_max_concurrent(4)
        path = str(self.root / 'data/results.h5')
        readers = [self.manager.submit(self.job('reader' + str(i), reads=[path])) for i in range(2)]
        self.ready(*readers)
        writer = self.manager.submit(self.job('writer', writes=[str(self.root / 'data')]))
        later = self.manager.submit(self.job('later-reader', reads=[path]))
        independent = self.manager.submit(self.job('independent'))
        self.ready(independent)
        self.assertEqual(writer.status, 'queued')
        self.assertIn('reader', writer.wait_reason)
        self.assertEqual(later.status, 'queued')
        for reader in readers:
            self.release(reader)
        self.ready(writer)
        self.assertEqual(later.status, 'queued')
        self.release(writer)
        self.ready(later)

    def test_cancelling_waiting_writer_unblocks_later_reader(self):
        path = str(self.root / 'shared.txt')
        reader = self.manager.submit(self.job('reader', reads=[path]))
        writer = self.manager.submit(self.job('writer', writes=[path]))
        later = self.manager.submit(self.job('later', reads=[path]))
        self.assertEqual(later.status, 'queued')
        self.manager.cancel(writer.id)
        self.ready(reader, later)

    def test_relative_aliases_and_same_panel_reserve_the_same_resource(self):
        first = self.manager.submit(self.job('first', cwd=str(self.root), writes=['out/../result.txt']))
        second = self.manager.submit(self.job('second', reads=[str(self.root / 'result.txt')]))
        self.assertEqual(second.status, 'queued')
        self.release(first)
        self.ready(second)
        another = self.manager.submit(self.job('second'))
        self.assertEqual(another.status, 'queued')
        self.assertIn('panel:second', another.wait_reason)

    def test_matlab_exclusive_lock_does_not_block_unrelated_work(self):
        first = self.manager.submit(self.job('matlab1', exclusive=['MATLAB/CPBASEX']))
        second = self.manager.submit(self.job('matlab2', exclusive=['MATLAB/CPBASEX']))
        other = self.manager.submit(self.job('pyabel'))
        self.ready(first, other)
        self.assertEqual(second.status, 'queued')
        self.release(first)
        self.ready(second)

    def test_pipeline_reserves_future_outputs_and_runs_stages_in_order(self):
        a, b = self.job('stage-a'), self.job('stage-b')
        output = str(self.root / 'checkpoint.h5')
        pipeline = self.manager.submit(Job(name='Pipeline', stages=[
            Stage('a', a.cmd), Stage('b', b.cmd, outputs=[output])]))
        wait_for(lambda: any(line.endswith('stage-a') for line in pipeline.log))
        consumer = self.manager.submit(self.job('consumer', reads=[output]))
        self.assertEqual(consumer.status, 'queued')
        self.assertEqual(pipeline.stages[1].status, 'queued')
        (self.root / 'stage-a.release').touch()
        wait_for(lambda: pipeline.stages[1].status == 'running')
        self.assertEqual(pipeline.stages[0].status, 'done')
        self.assertEqual(consumer.status, 'queued')
        (self.root / 'stage-b.release').touch()
        self.ready(consumer)
        self.assertEqual(pipeline.status, 'done')

    def test_clear_outputs_cannot_delete_files_reserved_by_jobs(self):
        from vmi_analysis.gui.panels.common import delete_outputs
        path = self.root / 'protected.txt'
        path.write_text('unchanged')
        reader = self.manager.submit(self.job('reader', reads=[str(path)]))
        with patch('vmi_analysis.gui.panels.common.MANAGER', self.manager), \
                patch('vmi_analysis.gui.panels.common.ui.notify') as notify:
            self.assertIsNone(delete_outputs([path]))
            self.assertEqual(path.read_text(), 'unchanged')
            notify.assert_called_once()
            self.release(reader)
            self.assertEqual(delete_outputs([path]), [])
            self.assertFalse(path.exists())

    def test_shutdown_cancels_all_running_and_queued_jobs(self):
        jobs = [self.manager.submit(self.job(str(i))) for i in range(5)]
        self.ready(*jobs[:3])
        processes = [j._proc for j in jobs[:3]]
        self.manager.shutdown()
        self.assertTrue(all(j.status == 'cancelled' for j in jobs))
        self.assertTrue(all(p.poll() is not None for p in processes))
        self.assertTrue(all(j.started is None for j in jobs[3:]))
        with self.assertRaises(RuntimeError):
            self.manager.submit(self.job('after-shutdown'))

    def test_process_failure_releases_the_file_reservation(self):
        path = str(self.root / 'shared.txt')
        first = self.manager.submit(Job(name='failure', cmd=[sys.executable, '-c', 'raise SystemExit(2)'], writes=[path]))
        next_job = self.manager.submit(self.job('next', writes=[path]))
        self.ready(next_job)
        self.assertEqual(first.status, 'failed')


if __name__ == '__main__':
    unittest.main()
