from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vmi_analysis.gui.panels.picker import _create_directory


class NewFolderTests(unittest.TestCase):
    def test_creates_one_child_with_spaces_and_unicode(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            target = _create_directory(parent, ' Analysis 数据 ')
            self.assertEqual(target, parent / 'Analysis 数据')
            self.assertTrue(target.is_dir())
            self.assertEqual(list(parent.iterdir()), [target])

    def test_existing_files_and_directories_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            (parent / 'existing').mkdir()
            file = parent / 'important.txt'
            file.write_text('keep this data')
            for name in ('existing', 'important.txt'):
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    _create_directory(parent, name)
            self.assertEqual(file.read_text(), 'keep this data')

    def test_rejects_paths_empty_names_and_windows_reserved_names(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            for name in ('', ' ', '.', '..', '../outside', 'a/b', 'a\\b', '/outside',
                         'C:\\outside', 'a:b', 'bad?', 'bad\x00', 'trailing.', 'CON', 'nul.txt', 'COM1'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    _create_directory(parent, name)
            self.assertEqual(list(parent.iterdir()), [])

    def test_filesystem_errors_propagate_for_the_dialog_to_report(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, 'mkdir', side_effect=PermissionError('read-only location')):
                with self.assertRaises(PermissionError):
                    _create_directory(Path(folder), 'new')
            self.assertFalse((Path(folder) / 'new').exists())


if __name__ == '__main__':
    unittest.main()
