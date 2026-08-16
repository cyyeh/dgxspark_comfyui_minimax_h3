import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import install_wizard


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_INSTALLER = ROOT / "deploy_from_scratch.sh"
CLONE_INSTALLER = ROOT / "deploy_to_new_spark.sh"


class ScratchInstallerDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.log = self.root / "wget.log"
        fake_wget = self.bin_dir / "wget"
        fake_wget.write_text(
            """#!/bin/bash
url="${@: -1}"
filename="${url##*/}"
if [ "${FAKE_WGET_NO_WRITE:-0}" != "1" ]; then
    printf '%s' "${FAKE_WGET_DATA:-}" >> "$filename"
fi
printf '%s\n' "$*" >> "$FAKE_WGET_LOG"
exit "${FAKE_WGET_EXIT:-0}"
"""
        )
        fake_wget.chmod(0o755)
        self.env = os.environ.copy()
        self.env.update(
            {
                "MODEL_DIR": str(self.model_dir),
                "FAKE_WGET_LOG": str(self.log),
                "PATH": f"{self.bin_dir}:{self.env['PATH']}",
            }
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_download(self, relative_path, expected_size, **environment):
        source = SCRATCH_INSTALLER.read_text()
        self.assertRegex(
            source,
            r'if \[ "\$\{BASH_SOURCE\[0\]\}" = "\$0" \]; then\s+main\s+fi',
            "the installer must be sourceable without running main",
        )
        self.assertIn("download_hf_file()", source)
        env = self.env.copy()
        env.update(environment)
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; download_hf_file "$2" "$3" "$4" "$5" || exit $?',
                "bash",
                str(SCRATCH_INSTALLER),
                "org/repo",
                "a" * 40,
                relative_path,
                str(expected_size),
            ],
            capture_output=True,
            text=True,
            env=env,
        )

    def test_complete_file_skips_wget(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"complete")

        result = self.run_download("model.bin", 8)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())

    def test_partial_file_resumes_once_and_then_skips(self):
        model = self.model_dir / "nested" / "model.bin"
        model.parent.mkdir()
        model.write_bytes(b"part")

        first = self.run_download("nested/model.bin", 7, FAKE_WGET_DATA="ial")
        second = self.run_download("nested/model.bin", 7, FAKE_WGET_DATA="unused")

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(model.read_bytes(), b"partial")
        calls = self.log.read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn("--continue", calls[0])
        self.assertIn("--tries=10", calls[0])
        self.assertNotIn("--tries=0", calls[0])
        self.assertIn("/resolve/" + "a" * 40 + "/nested/model.bin", calls[0])

    def test_partial_symlink_is_materialized_before_resume(self):
        target = self.root / "legacy.bin"
        target.write_bytes(b"part")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)

        result = self.run_download("model.bin", 7, FAKE_WGET_DATA="ial")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(model.is_symlink())
        self.assertEqual(model.read_bytes(), b"partial")
        self.assertEqual(target.read_bytes(), b"part")

    def test_complete_symlink_is_materialized_without_wget(self):
        target = self.root / "legacy.bin"
        target.write_bytes(b"complete")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)

        result = self.run_download("model.bin", 8)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(model.is_symlink())
        self.assertEqual(model.read_bytes(), b"complete")
        self.assertFalse(self.log.exists())

    def test_broken_symlink_is_quarantined_before_download(self):
        model = self.model_dir / "model.bin"
        model.symlink_to(self.root / "missing.bin")

        result = self.run_download("model.bin", 5, FAKE_WGET_DATA="fresh")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(model.read_bytes(), b"fresh")
        self.assertTrue((self.model_dir / "model.bin.invalid").is_symlink())

    def test_oversized_file_is_quarantined_before_download(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"oversized")

        result = self.run_download("model.bin", 7, FAKE_WGET_DATA="correct")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(model.read_bytes(), b"correct")
        self.assertEqual(
            (self.model_dir / "model.bin.invalid").read_bytes(), b"oversized"
        )

    def test_failed_wget_keeps_partial_file_and_returns_retry_guidance(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"part")

        result = self.run_download("model.bin", 7, FAKE_WGET_EXIT="1")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(model.read_bytes(), b"part")
        self.assertIn("重新运行", result.stdout + result.stderr)
        self.assertIn("续传", result.stdout + result.stderr)

    def test_directory_creation_failure_never_invokes_wget(self):
        blocker = self.root / "blocker"
        blocker.write_text("not a directory")

        result = self.run_download(
            "model.bin",
            5,
            MODEL_DIR=str(blocker / "models"),
            FAKE_WGET_DATA="fresh",
            FAKE_WGET_NO_WRITE="1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_failed_quarantine_never_invokes_wget(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"oversized")
        fake_mv = self.bin_dir / "mv"
        fake_mv.write_text("#!/bin/bash\nexit 1\n")
        fake_mv.chmod(0o755)

        result = self.run_download(
            "model.bin",
            7,
            FAKE_WGET_DATA="correct",
            FAKE_WGET_NO_WRITE="1",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(model.read_bytes(), b"oversized")
        self.assertFalse(self.log.exists())

    def test_materialization_copy_failure_keeps_symlink_and_cleans_temp(self):
        target = self.root / "legacy.bin"
        target.write_bytes(b"part")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)
        fake_cp = self.bin_dir / "cp"
        fake_cp.write_text("#!/bin/bash\nexit 1\n")
        fake_cp.chmod(0o755)

        result = self.run_download("model.bin", 7, FAKE_WGET_DATA="ial")

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(model.is_symlink())
        self.assertEqual(target.read_bytes(), b"part")
        self.assertEqual(list(self.model_dir.glob("model.bin.materializing.*")), [])
        self.assertFalse(self.log.exists())

    def test_wrong_post_download_size_is_rejected(self):
        model = self.model_dir / "model.bin"

        result = self.run_download("model.bin", 7, FAKE_WGET_DATA="short")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(model.read_bytes(), b"short")
        self.assertIn("大小不符", result.stdout + result.stderr)

    def test_product_shell_scripts_have_no_modelscope_or_unpinned_downloads(self):
        combined = SCRATCH_INSTALLER.read_text() + CLONE_INSTALLER.read_text()
        lowered = combined.lower()
        self.assertNotIn("modelscope", lowered)
        self.assertNotRegex(lowered, r"resolve/(?:main|master)/")
        self.assertNotIn("huggingface-cli download", lowered)
        self.assertNotIn("huggingface_hub.cli.hf download", lowered)
        self.assertNotIn("--tries=0", lowered)

    def test_shell_manifest_matches_interactive_installer(self):
        source = SCRATCH_INSTALLER.read_text()
        self.assertIn(install_wizard.HF_BASE_REPO, source)
        self.assertIn(install_wizard.HF_BASE_REVISION, source)
        self.assertIn(install_wizard.HF_EXTRA_REPO, source)
        self.assertIn(install_wizard.HF_EXTRA_REVISION, source)
        for relative_path, expected_size in {
            **install_wizard.HF_BASE_MANIFEST,
            **install_wizard.HF_EXTRA_MANIFEST,
        }.items():
            with self.subTest(relative_path=relative_path):
                self.assertIn(f'"{relative_path}" {expected_size}', source)


class CloneInstallerTests(unittest.TestCase):
    def test_clone_dereferences_model_symlinks_without_separate_cache_transfer(self):
        source = CLONE_INSTALLER.read_text().lower()
        self.assertIn("rsync -al", source)
        self.assertNotIn("modelscope", source)
        self.assertNotIn("ln -sf", source)


if __name__ == "__main__":
    unittest.main()
