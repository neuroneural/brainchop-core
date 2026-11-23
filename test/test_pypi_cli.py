#!/usr/bin/env python3
"""CLI compatibility contracts; run with standard-library Python, no ML packages."""

from contextlib import redirect_stderr, redirect_stdout
import gzip
import json
import importlib.util
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging'))
from brainchop_cli import cli, native


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='brainchop-cli-test-')
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / 'scan with spaces.nii.gz'
        self.source.write_bytes(b'test input')

    def args(self, *arguments):
        return cli.get_parser().parse_args([str(self.source), *arguments])

    def main(self, *arguments):
        with patch.object(sys, 'argv', ['brainchop', *map(str, arguments)]):
            cli.main()

    def test_all_historical_options_are_parsed(self):
        args = self.args('-o', 'labels.nii.gz', '-m', 'subcortical', '--ct', '--comply',
                         '--inverse-conform', '--export-classes', '--mask', 'mask.nii.gz',
                         '--border', '3', '--batch-size', '4', '--beam', '2', '--no-optimize')
        self.assertEqual((args.model, args.output, args.border), ('subcortical', 'labels.nii.gz', 3))
        self.assertTrue(args.ct and args.comply and args.inverse_conform and args.export_classes)
        self.assertEqual(self.args('--crop').crop, 2.0)
        self.assertEqual(self.args('--mask').mask, 'mask.nii.gz')
        self.assertEqual(self.args('--custom', 'weights').custom, 'weights')
        self.assertTrue(cli.get_parser().parse_args(['-l']).list)
        self.assertTrue(cli.get_parser().parse_args(['-u']).update)

    def test_default_is_supported_embedded_model(self):
        self.assertEqual(self.args().model, 'mindmap')
        self.assertEqual(set(native.NATIVE_MODELS), {'mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap'})

    def test_default_and_conformed_output_spaces(self):
        for model in ('mindmap-fast', 'mindmap', 'mindsnap'):
            default = native.command(self.args(), model, 'binary', self.source, 'out.nii.gz')
            inverse = native.command(self.args('--inverse-conform'), model, 'binary', self.source,
                                     'out.nii.gz')
            conformed = native.command(self.args('--save-conform'), model, 'binary', self.source,
                                       'out.nii.gz')
            self.assertNotIn('--save-conform', default)
            self.assertNotIn('--save-conform', inverse)
            self.assertNotIn('--inverse-conform', inverse)
            self.assertIn('--save-conform', conformed)
        self.assertNotIn('--save-conform', native.command(self.args(), 'mindgrab', 'binary',
                                                         self.source, 'out.nii.gz'))

    def test_output_names_match_the_historical_cli(self):
        args = self.args()
        self.assertEqual(cli._output_path(args, 'mindmap', 0, 'scan.nii.gz'), 'scan_mindmap_1.nii.gz')
        self.assertEqual(cli._output_path(self.args('-o', 'chosen.nii'), 'mindmap', 0, 'scan.nii.gz'),
                         'chosen.nii')
        args.input.append('second.nii.gz')
        args.output = 'ignored-for-multiple-inputs.nii.gz'
        self.assertEqual(cli._output_path(args, 'subcortical', 1, 'second.nii.gz'),
                         'second_subcortical_2.nii.gz')

    def test_mask_crop_and_preprocessing_are_forwarded_as_distinct_arguments(self):
        args = self.args('--crop', '2.5', '--mask', 'mask with spaces.nii.gz', '--border', '3',
                         '--ct', '--comply', '--threads', '2')
        native.validate_options(args, 'mindgrab')
        cmd = native.command(args, 'mindgrab', 'binary', self.source, 'output with spaces.nii.gz')
        self.assertEqual(cmd, ['binary', '-o', 'output with spaces.nii.gz', '--ct', '--comply',
                               '--crop', '2.5', '--mask', 'mask with spaces.nii.gz', '--border', '3',
                               '-cpu', '2', str(self.source)])

    def test_no_turbo_is_forwarded(self):
        cmd = native.command(self.args('--no-turbo'), 'mindmap', 'binary', self.source, 'o.nii.gz')
        self.assertIn('--no-turbo', cmd)
        self.assertNotIn('--no-turbo', native.command(self.args(), 'mindmap', 'binary',
                                                      self.source, 'o.nii.gz'))

    def test_different_legacy_checkpoints_are_rejected(self):
        for name in ('subcortical', 'DKatlas', 'tissue_fast', 'robust_tissue', 'aparc50'):
            with self.subTest(name=name), patch.object(native.subprocess, 'run') as launch, redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                self.main(self.source, '-m', name)
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_skull_strip_overrides_model_and_custom(self):
        with patch.object(native, 'find_executable', return_value='binary') as find, \
             patch.object(native.subprocess, 'run'), redirect_stdout(StringIO()):
            self.main(self.source, '-m', 'retired', '--custom', 'ignored', '--skull-strip')
        find.assert_called_once_with('mindgrab')

    def test_retired_models_and_custom_fail_before_launch(self):
        for extra in (['-m', 'tissue_fast'], ['-m', 'robust_tissue'], ['-m', 'aparc50'],
                      ['--custom', 'weights']):
            with self.subTest(extra=extra), patch.object(native.subprocess, 'run') as launch, \
                 redirect_stderr(StringIO()), self.assertRaises(SystemExit) as error:
                self.main(self.source, *extra)
            self.assertEqual(error.exception.code, 2)
            launch.assert_not_called()

    def test_incompatible_options_are_rejected_before_launch(self):
        # Model/flag conflicts are the native CLI's to reject; these are the adapter's.
        cases = [('mindmap', ['--batch-size', '0']),
                 ('mindgrab', ['--save-conform']), ('mindmap', ['--threads', '-1']),
                 ('mindmap', ['--threads', '2', '--backend', 'cuda'])]
        for model, extra in cases:
            with self.subTest(model=model, extra=extra), self.assertRaises(ValueError):
                native.validate_options(self.args(*extra), model)

    def test_entire_batch_is_checked_before_launch(self):
        args = self.args(str(Path(self.directory.name) / 'missing.nii.gz'))
        with patch.object(native.subprocess, 'run') as launch, self.assertRaises(ValueError):
            native.run(args, 'mindmap', cli._output_path)
        launch.assert_not_called()

    def test_legacy_performance_flags_are_accepted_with_explanation(self):
        with patch.object(native, 'find_executable', return_value='binary'), \
             patch.object(native.subprocess, 'run') as launch, \
             redirect_stderr(StringIO()) as stderr, redirect_stdout(StringIO()):
            self.main(self.source, '--beam', '2', '--batch-size', '4', '--no-optimize')
        self.assertIn('compatibility options', stderr.getvalue())
        self.assertNotIn('--beam', launch.call_args.args[0])
        self.assertNotIn('--batch-size', launch.call_args.args[0])

    def test_multiple_inputs_are_run_sequentially(self):
        second = Path(self.directory.name) / 'second.nii'
        second.write_bytes(b'another input')
        args = self.args(str(second))
        with patch.object(native, 'find_executable', return_value='binary'), \
             patch.object(native.subprocess, 'run') as launch, redirect_stdout(StringIO()):
            native.run(args, 'mindmap', cli._output_path)
        self.assertEqual(launch.call_count, 2)
        self.assertEqual(launch.call_args_list[1].args[0][2], 'second_mindmap_2.nii.gz')

    def test_subprocess_failures_preserve_exit_status_and_do_not_fallback(self):
        for code, expected in ((7, 7), (-9, 137)):
            with patch.object(native, 'find_executable', return_value='binary'), \
                 patch.object(native.subprocess, 'run',
                              side_effect=subprocess.CalledProcessError(code, ['binary'])) as launch, \
                 redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
                native.run(self.args(), 'mindmap', cli._output_path)
            self.assertEqual(error.exception.code, expected)
            self.assertEqual(launch.call_count, 1)

    def test_optional_cpu_instructions_are_guarded_before_launch(self):
        package=Path(self.directory.name)
        binaries=package/'bin'; binaries.mkdir(); (binaries/'cpu-fallback').mkdir()
        suffix='.exe' if os.name=='nt' else ''
        primary=binaries/('brainchop-mindmap'+suffix); primary.write_bytes(b'primary')
        portable=binaries/'cpu-fallback'/primary.name; portable.write_bytes(b'portable')
        (package/'build.json').write_text(json.dumps(dict(cpu_target='avx2-fma',options=['HIP=1'],executables={primary.name:dict(linkage=['libc.so.6'])})))
        with patch.object(native,'__file__',str(package/'native.py')), patch.object(native.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'cpu\tavx2-fma\t1\n','')):
            self.assertEqual(native.find_executable('mindmap'),str(primary))
        with patch.object(native,'__file__',str(package/'native.py')), patch.object(native.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'','')), self.assertRaisesRegex(ValueError,'requires AVX2 and FMA'):
            native.find_executable('mindmap')
        with patch.object(native,'__file__',str(package/'native.py')), patch.object(native.subprocess,'run',side_effect=subprocess.TimeoutExpired('probe',10)), self.assertRaisesRegex(ValueError,'cannot be verified'):
            native.find_executable('mindmap')

    def test_keyboard_interrupt_has_clean_exit_status(self):
        with patch.object(native, 'find_executable', return_value='binary'), patch.object(native.subprocess,'run',side_effect=KeyboardInterrupt), redirect_stdout(StringIO()), self.assertRaises(SystemExit) as error:
            native.run(self.args(),'mindmap',cli._output_path)
        self.assertEqual(error.exception.code,130)

    def test_missing_binary_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, 'missing from this installation'):
            native.find_executable('mindmap')

    def test_exported_raw_maps_retain_legacy_names_and_bytes(self):
        output = str(Path(self.directory.name) / 'labels.nii.gz')
        for index in range(2):
            Path(output[:-7] + f'_c{index}.nii.gz').write_bytes(gzip.compress(b'float-map-' + bytes([index])))
        native.export_legacy_filenames(output, 'mindgrab')
        for index in range(2):
            self.assertEqual(Path(output[:-7] + f'_c{index}.nii').read_bytes(),
                             b'float-map-' + bytes([index]))
            self.assertFalse(Path(output[:-7] + f'_c{index}.nii.gz').exists())

    def test_informational_commands_do_not_require_a_binary_or_ml_runtime(self):
        for option in ('--list', '--update', '--help'):
            result = subprocess.run([sys.executable, '-S', '-m', 'brainchop_cli', option],
                                    env={**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'packaging')},
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
