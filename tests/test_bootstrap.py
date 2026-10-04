"""Bootstrap checks use mocks only: no installs, venv creation, or model runs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("portable_bootstrap_under_test", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_help_never_prepares_environment(self):
        with patch.object(bootstrap, "ensure_environment") as prepare, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                bootstrap.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        prepare.assert_not_called()

    def test_invalid_arguments_do_not_prepare_environment(self):
        for arguments in (["--budget-seconds", "nan"], ["--budget-seconds", "7201"],
                          ["--max-mib", "0"], ["--max-mib", "129"],
                          ["--seed-offset", "-1"], ["--seed-offset", "1000000001"],
                          ["--condition", "certified-clean"]):
            with self.subTest(arguments=arguments), patch.object(bootstrap, "ensure_environment") as prepare:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    bootstrap.main(arguments)
                prepare.assert_not_called()

    def test_defaults_and_declared_maximum(self):
        defaults = bootstrap.parser().parse_args([])
        self.assertEqual((defaults.budget_seconds, defaults.max_mib, defaults.condition), (600, 16, "unspecified"))
        parsed = bootstrap.parser().parse_args(["--budget-seconds", "7200", "--max-mib", "128"])
        self.assertEqual((parsed.budget_seconds, parsed.max_mib), (7200, 128))
        self.assertEqual(bootstrap.parser().parse_args(["--seed-offset", "1000000000"]).seed_offset, 1_000_000_000)

    def test_platform_interpreter_paths_preserve_spaces(self):
        root = Path("checkout with spaces")
        self.assertEqual(bootstrap.venv_python(root, platform="nt"), root / ".venv/Scripts/python.exe")
        self.assertEqual(bootstrap.venv_python(root, platform="posix"), root / ".venv/bin/python")

    def test_setup_only_does_not_dispatch_benchmark(self):
        with patch.object(bootstrap, "ensure_environment", return_value=Path("fake-python")), \
             patch.object(bootstrap, "_run_benchmark") as run, \
             patch.object(bootstrap.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(bootstrap.main(["--setup-only"]), 0)
        run.assert_not_called()
        child.assert_not_called()

    def test_reentry_uses_owned_interpreter_isolation_and_argument_list(self):
        root = Path("checkout with spaces").resolve()
        python = root / ".venv/Scripts/python.exe"
        arguments = ["--preset", "idle-primary", "--output", "result with spaces"]
        with patch.object(bootstrap, "ROOT", root), \
             patch.object(bootstrap, "ensure_environment", return_value=python), \
             patch.object(bootstrap, "_already_isolated", return_value=False), \
             patch.object(bootstrap.subprocess, "run", return_value=MagicMock(returncode=7)) as run:
            self.assertEqual(bootstrap.main(arguments), 7)
        self.assertEqual(run.call_args.args[0], [str(python), "-I", "-B", str(root / "run_experiment.py"), *arguments])
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_owned_isolated_environment_dispatches_once(self):
        with patch.object(bootstrap, "ensure_environment", return_value=Path("fake-python")), \
             patch.object(bootstrap, "_already_isolated", return_value=True), \
             patch.object(bootstrap, "_run_benchmark", return_value=0) as run:
            self.assertEqual(bootstrap.main(["--condition", "busy"]), 0)
        run.assert_called_once_with(bootstrap.ROOT, ["--condition", "busy"])

    def test_old_python_fails_before_setup(self):
        with patch.object(bootstrap.sys, "version_info", (3, 10, 9)), \
             patch.object(bootstrap, "ensure_environment") as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(bootstrap.main([]), 2)
        prepare.assert_not_called()

    def test_setup_failure_is_explicit_without_retry(self):
        with patch.object(bootstrap, "ensure_environment", side_effect=bootstrap.BootstrapError("network failure")) as prepare, \
             patch.object(bootstrap, "_run_benchmark") as run, contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(bootstrap.main([]), 2)
        self.assertIn("network failure", errors.getvalue())
        self.assertEqual(prepare.call_count, 1)
        run.assert_not_called()

    def test_matching_dependency_does_not_install(self):
        root = Path("fake-root").resolve()
        with patch.object(Path, "exists", return_value=True), \
             patch.object(bootstrap, "_check_layout", return_value=Path("fake-python")), \
             patch.object(bootstrap, "_probe", return_value={}), \
             patch.object(bootstrap, "_dependency_ready", return_value=True), \
             patch.object(bootstrap, "_setup_command") as command:
            self.assertEqual(bootstrap.ensure_environment(root), Path("fake-python"))
        command.assert_not_called()

    def test_missing_dependency_uses_hash_locked_wheel_only(self):
        root = Path("fake-root with spaces").resolve()
        python = root / ".venv/bin/python"
        with patch.object(Path, "exists", return_value=True), patch.object(Path, "is_file", return_value=True), \
             patch.object(bootstrap, "_check_layout", return_value=python), \
             patch.object(bootstrap, "_probe", return_value={}) as probe, \
             patch.object(bootstrap, "_dependency_ready", side_effect=[False, True]), \
             patch.object(bootstrap, "_setup_command") as command, contextlib.redirect_stdout(io.StringIO()):
            bootstrap.ensure_environment(root)
        command.assert_called_once()
        argv = command.call_args.args[0]
        self.assertEqual(argv[:6], [str(python), "-I", "-B", "-m", "pip", "--isolated"])
        for flag in ("--require-hashes", "--only-binary=:all:", "--no-deps", "--force-reinstall"):
            self.assertIn(flag, argv)
        self.assertEqual(argv[-2:], ["-r", str(root / "requirements.lock")])
        self.assertEqual(probe.call_count, 2)

    def test_dependency_must_be_exact_and_inside_owned_environment(self):
        root = Path("fake-root").resolve()
        receipt = {"dill_version": "0.4.1", "dill_import_version": "0.4.1",
                   "dill_path": str(root / ".venv/lib/dill/__init__.py")}
        self.assertTrue(bootstrap._dependency_ready(receipt, root))
        self.assertFalse(bootstrap._dependency_ready(dict(receipt, dill_path=str(root / "usersite/dill.py")), root))
        self.assertFalse(bootstrap._dependency_ready(dict(receipt, dill_import_version="0.4.0"), root))

    def test_probe_rejects_external_prefix(self):
        root = Path("fake-root").resolve()
        receipt = {"prefix": str(root / "other-venv"), "base_prefix": str(root / "base"),
                   "python": [3, 11, 0], "isolated": 1, "dont_write_bytecode": True, "user_site": False}
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(receipt))
        with patch.object(bootstrap, "_setup_command", return_value=result), self.assertRaises(bootstrap.BootstrapError):
            bootstrap._probe(Path("fake-python"), root)

    def test_setup_timeout_becomes_clear_error(self):
        with patch.object(bootstrap.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 300)), \
             self.assertRaisesRegex(bootstrap.BootstrapError, "300 seconds"):
            bootstrap._setup_command(["fake-python"])

    def test_system_site_environment_is_not_modified(self):
        with patch.object(Path, "is_symlink", return_value=False), \
             patch.object(Path, "is_junction", return_value=False, create=True), \
             patch.object(Path, "is_dir", return_value=True), patch.object(Path, "is_file", return_value=True), \
             patch.object(Path, "read_text", return_value="include-system-site-packages = true\n"), \
             self.assertRaisesRegex(bootstrap.BootstrapError, "system-site-packages"):
            bootstrap._check_layout(Path("fake-root"))

    def test_dangling_junction_is_rejected_before_venv_creation(self):
        with patch.object(Path, "exists", return_value=False), \
             patch.object(Path, "is_symlink", return_value=False), \
             patch.object(Path, "is_junction", return_value=True, create=True), \
             patch.object(bootstrap, "_setup_command") as command, \
             self.assertRaisesRegex(bootstrap.BootstrapError, "symlink or junction"):
            bootstrap.ensure_environment(Path("fake-root"))
        command.assert_not_called()

    def test_linux_venv_failure_explains_manual_package_requirement(self):
        root = Path("fake-root")
        with patch.object(bootstrap.os, "name", "posix"), \
             patch.object(bootstrap.sys, "version_info", (3, 12, 3)), \
             patch.object(bootstrap, "_reject_environment_link"), \
             patch.object(Path, "exists", return_value=False), \
             patch.object(bootstrap, "_setup_command", side_effect=bootstrap.BootstrapError("No module named ensurepip")) as command, \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(bootstrap.BootstrapError) as failure:
            bootstrap.ensure_environment(root)
        message = str(failure.exception)
        self.assertIn("python3-venv", message)
        self.assertIn("python3.12-venv", message)
        self.assertIn("No system packages were installed", message)
        self.assertIn("partial .venv", message)
        command.assert_called_once()


if __name__ == "__main__":
    unittest.main()
