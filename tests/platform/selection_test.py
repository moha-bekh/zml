"""Linux integration checks; namespace changes affect only child processes."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from python.runfiles import runfiles


class PlatformSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = runfiles.Create()
        cls.binary = files.Rlocation(sys.argv[1])
        workspace = sys.argv[1].split('/')[0]
        cls.plugin = Path(files.Rlocation(
            workspace + '/platforms/cpu/sandbox/lib/libzml_cpu.so')).resolve()
        if not cls.plugin.is_file():
            raise RuntimeError('CPU plugin runfile not found')
        cls.bwrap = shutil.which('bwrap')

    def isolated(self):
        if not self.bwrap:
            self.skipTest('bubblewrap is required for isolation cases')
        prefix = [self.bwrap, '--ro-bind', '/', '/', '--dev', '/dev',
                  '--unshare-user', '--die-with-parent']
        check = subprocess.run(prefix + [sys.executable, '-c', 'pass'],
                               capture_output=True, timeout=10)
        if check.returncode:
            self.skipTest('user namespaces unavailable: ' + check.stderr.decode())
        return prefix

    def run_probe(self, mode, prefix=()):
        result = subprocess.run([*prefix, self.binary, '--mode=' + mode],
                                capture_output=True, text=True, timeout=90)
        self.assertIn('probe_started mode=' + mode, result.stderr)
        return result

    def successful(self, result, backend):
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('selected_backend=' + backend, result.stderr)
        self.assertIn('result=42', result.stderr)

    def test_explicit_cpu(self):
        self.successful(self.run_probe('cpu'), 'cpu')

    def test_auto_without_devices(self):
        self.successful(self.run_probe('auto', self.isolated()), 'cpu')

    def test_cuda_control(self):
        capabilities = self.run_probe('capabilities')
        self.assertEqual(capabilities.returncode, 0, capabilities.stderr)
        if 'cuda_enabled=true' not in capabilities.stderr:
            self.skipTest('CUDA backend not enabled in this build')
        if not any(Path(path).exists() for path in ('/dev/nvidiactl', '/dev/dxg')):
            self.skipTest('no NVIDIA device exposed to the test')
        self.successful(self.run_probe('cuda'), 'cuda')

    def test_explicit_cuda_without_devices(self):
        capabilities = self.run_probe('capabilities')
        self.assertEqual(capabilities.returncode, 0, capabilities.stderr)
        if 'cuda_enabled=true' not in capabilities.stderr:
            self.skipTest('CUDA backend not enabled in this build')
        result = self.run_probe('cuda', self.isolated())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('error: Unavailable', result.stderr)
        self.assertNotIn('selected_backend=', result.stderr)

    def test_plugin_failures(self):
        prefix = self.isolated()
        if os.geteuid() == 0:
            self.skipTest('permission denial must run as an unprivileged user')
        with tempfile.TemporaryDirectory(dir=os.environ['TEST_TMPDIR']) as temp:
            invalid = Path(temp) / 'invalid.so'
            invalid.write_bytes(b'not an ELF library\n')
            denied = Path(temp) / 'denied.so'
            denied.write_bytes(b'permission fixture\n')
            denied.chmod(0)
            cases = (
                ('missing', ['--tmpfs', str(self.plugin.parent)], 'No such file'),
                ('corrupt', ['--ro-bind', str(invalid), str(self.plugin)], 'file too short'),
                ('denied', ['--ro-bind', str(denied), str(self.plugin)], 'Permission denied'),
            )
            try:
                for name, mounts, cause in cases:
                    for mode in ('cpu', 'auto'):
                        with self.subTest(case=name, mode=mode):
                            result = self.run_probe(mode, prefix + mounts)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn('Unable to dlopen plugin', result.stderr)
                            self.assertIn(cause, result.stderr)
                            self.assertIn('error: ' + ('FileNotFound' if mode == 'cpu'
                                                      else 'Unavailable'), result.stderr)
                            self.assertNotIn('selected_backend=', result.stderr)
            finally:
                denied.chmod(0o600)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
