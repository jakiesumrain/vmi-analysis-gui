"""Run/Stop button states, reload recovery, and real subprocess cancellation."""

import sys
import time
import unittest
from unittest.mock import Mock, patch

from nicegui import Client, ui

from vmi_analysis.gui import theme
from vmi_analysis.gui.jobs import Job, JobManager
from vmi_analysis.gui.panels.common import JobView


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Timed out waiting for subprocess state")


class ButtonFixture(unittest.TestCase):
    def make_view(self, manager, on_run, job_name="Centroiding", label="Run"):
        callbacks = []
        def button(text="Run", **kwargs):
            callbacks.append(kwargs["on_click"])
            return ui.button(text, **kwargs)
        client = Client(ui.page("/_test_run_stop"))
        self.addCleanup(client.delete)
        with client, patch.object(theme, "run_button", button):
            view = JobView(manager, job_name=job_name)
            view.run_button(label, on_click=on_run)
            view.place_console()
        return view, callbacks[0]


class RunStopButtonTests(ButtonFixture):
    def setUp(self):
        self.manager = Mock()
        self.manager.jobs.return_value = []
        def cancel(job_id):
            self.job._cancel_requested = True
            if self.job.status == "queued":
                self.job.status = "cancelled"
        self.manager.cancel.side_effect = cancel

    def test_run_stop_stopping_and_restart_use_one_button(self):
        run = Mock()
        def start():
            run()
            self.job = Job(name="Centroiding", status="running")
            view.attach(self.job)
        view, click = self.make_view(self.manager, start)
        button = view.action_button
        self.assertEqual(button.text, "Run")
        click()
        self.assertIs(view.action_button, button)
        self.assertEqual(button.text, "Stop")
        self.assertEqual(button._props["color"], "negative")
        click()
        self.manager.cancel.assert_called_once_with(self.job.id)
        self.assertEqual(button.text, "Stopping…")
        self.assertFalse(button.enabled)
        click()  # even duplicate queued events cannot submit/cancel again
        self.manager.cancel.assert_called_once()
        self.assertEqual(run.call_count, 1)
        self.job.status = "cancelled"
        view.refresh()
        self.assertEqual(button.text, "Run")
        self.assertTrue(button.enabled)
        click()
        self.assertEqual(run.call_count, 2)

    def test_queued_button_cancels_queued_job(self):
        view, click = self.make_view(self.manager, Mock())
        self.job = Job(name="Centroiding", status="queued")
        view.attach(self.job)
        self.assertEqual(view.action_button.text, "Cancel queued")
        click()
        self.assertEqual(self.job.status, "cancelled")
        self.assertEqual(view.action_button.text, "Run")

    def test_success_and_failure_restore_original_pipeline_label(self):
        for status in ("done", "failed", "cancelled"):
            with self.subTest(status=status):
                view, _ = self.make_view(self.manager, Mock(), "Pipeline", "Run pipeline")
                job = Job(name="Pipeline", status="running")
                view.attach(job)
                job.status = status
                view.refresh()
                self.assertEqual(view.action_button.text, "Run pipeline")
                self.assertTrue(view.action_button.enabled)

    def test_stale_stop_click_does_not_start_new_job(self):
        run = Mock()
        view, click = self.make_view(self.manager, run)
        self.job = Job(name="Centroiding", status="running")
        view.attach(self.job)
        self.job.status = "done"  # finishes just before Stop is clicked
        click()
        run.assert_not_called()
        self.assertEqual(view.action_button.text, "Run")

    def test_reload_restores_only_this_panels_running_job(self):
        self.job = Job(name="Centroiding", status="running")
        self.job.log.append("still processing")
        self.manager.jobs.return_value = [Job(name="Other", status="running"), self.job]
        view, click = self.make_view(self.manager, Mock())
        self.assertIs(view.job, self.job)
        self.assertEqual(view.action_button.text, "Stop")
        self.assertEqual(view._log_cursor, 1)
        click()
        self.manager.cancel.assert_called_once_with(self.job.id)
        other_view, _ = self.make_view(self.manager, Mock(), "Dark frame")
        self.assertIsNone(other_view.job)
        self.assertEqual(other_view.action_button.text, "Run")


class RealStopButtonTests(ButtonFixture):
    def setUp(self):
        self.manager = JobManager(max_concurrent=1)
        self.addCleanup(self.manager.shutdown)

    def sleeping_job(self, name):
        return Job(name=name, cmd=[sys.executable, "-c", "import time; time.sleep(60)"])

    def test_stop_button_terminates_real_subprocess(self):
        def start():
            view.attach(self.manager.submit(self.sleeping_job("Centroiding")))
        view, click = self.make_view(self.manager, start)
        click()
        wait_for(lambda: view.job._proc is not None)
        proc = view.job._proc
        self.assertEqual(view.action_button.text, "Stop")
        click()
        wait_for(lambda: view.job.status == "cancelled")
        view.refresh()
        self.assertIsNotNone(proc.poll())
        self.assertEqual(view.action_button.text, "Run")

    def test_cancel_queued_leaves_other_panels_running_process_alone(self):
        running = self.manager.submit(self.sleeping_job("Other panel"))
        wait_for(lambda: running._proc is not None)
        queued = self.manager.submit(self.sleeping_job("Centroiding"))
        view, click = self.make_view(self.manager, Mock())
        self.assertIs(view.job, queued)
        self.assertEqual(view.action_button.text, "Cancel queued")
        click()
        self.assertEqual(queued.status, "cancelled")
        self.assertIsNone(queued.started)
        self.assertIsNone(running._proc.poll())

    def test_stopping_one_concurrent_panel_leaves_the_other_running(self):
        self.manager.set_max_concurrent(2)
        first = self.manager.submit(self.sleeping_job("Centroiding"))
        second = self.manager.submit(self.sleeping_job("Dark frame"))
        wait_for(lambda: first._proc is not None and second._proc is not None)
        view, click = self.make_view(self.manager, Mock(), "Centroiding")
        other, _ = self.make_view(self.manager, Mock(), "Dark frame")
        self.assertEqual(view.action_button.text, "Stop")
        self.assertEqual(other.action_button.text, "Stop")
        click()
        wait_for(lambda: first.status == "cancelled")
        self.assertEqual(second.status, "running")
        self.assertIsNone(second._proc.poll())
        other.refresh()
        self.assertEqual(other.action_button.text, "Stop")


if __name__ == "__main__":
    unittest.main()
