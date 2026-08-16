import tempfile
import unittest
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import install_wizard


class StepComfyUIAndNodesTests(unittest.TestCase):
    def test_copies_nested_sol_attn_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            comfy_dir = root / "install" / "comfy" / "ComfyUI"
            cn_dir = comfy_dir / "custom_nodes"
            source = root / "keys-heretic"
            nested_source = (
                source
                / "vendor"
                / "ComfyUI_sol-attn_Blackwell"
                / "sol_attn"
                / "triton_ref"
                / "fwd.py"
            )
            nested_source.parent.mkdir(parents=True)
            nested_source.write_text("TRITON_KERNEL = True\n")
            (source / ".git").mkdir()
            (comfy_dir / ".git").mkdir(parents=True)

            install_wizard.cfg = SimpleNamespace(
                comfy_dir=comfy_dir,
                cn_dir=cn_dir,
                workflows_dir=root / "install" / "comfy" / "workflows",
                python=root / "venv" / "bin" / "python3",
            )

            real_path = Path

            def redirected_path(value):
                if value == "/tmp/keys-heretic-tmp":
                    return source
                return real_path(value)

            with (
                mock.patch.object(install_wizard, "Path", side_effect=redirected_path),
                mock.patch.object(install_wizard, "run"),
            ):
                try:
                    install_wizard.step_comfyui_and_nodes()
                except IsADirectoryError as exc:
                    self.fail(f"nested custom-node directories must be copied: {exc}")

            copied = (
                cn_dir
                / "ComfyUI_sol-attn_Blackwell"
                / "sol_attn"
                / "triton_ref"
                / "fwd.py"
            )
            self.assertEqual(copied.read_text(), "TRITON_KERNEL = True\n")


class HuggingFaceDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        install_wizard.cfg = SimpleNamespace(model_dir=self.model_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def download(self, manifest):
        self.assertTrue(
            hasattr(install_wizard, "download_hf_group"),
            "download_hf_group must be implemented",
        )
        return install_wizard.download_hf_group(
            "org/repo", "a" * 40, manifest, "测试模型"
        )

    def test_complete_manifest_skips_prompt_and_download(self):
        model = self.model_dir / "text_encoders" / "model.bin"
        model.parent.mkdir(parents=True)
        model.write_bytes(b"complete")

        with (
            mock.patch.object(install_wizard, "ask") as ask,
            mock.patch.object(install_wizard, "run") as run,
        ):
            self.download({"text_encoders/model.bin": 8})

        ask.assert_not_called()
        run.assert_not_called()

    def test_mixed_manifest_only_continues_partial_file_and_skips_on_retry(self):
        complete = self.model_dir / "complete.bin"
        partial = self.model_dir / "partial.bin"
        complete.write_bytes(b"done")
        partial.write_bytes(b"part")

        def finish_download(command, **kwargs):
            self.assertIn("--continue", command)
            self.assertIn("--tries=10", command)
            self.assertNotIn("--tries=0", command)
            self.assertNotIn("complete.bin", command[-1])
            self.assertTrue(command[-1].endswith("/partial.bin"))
            self.assertIsNone(kwargs["timeout"])
            self.assertFalse(kwargs["check"])
            with partial.open("ab") as output:
                output.write(b"ial")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(install_wizard, "ask", return_value=True) as ask,
            mock.patch.object(
                install_wizard, "run", side_effect=finish_download
            ) as run,
        ):
            manifest = {"complete.bin": 4, "partial.bin": 7}
            self.download(manifest)
            self.download(manifest)

        self.assertEqual(run.call_count, 1)
        self.assertEqual(ask.call_count, 1)

    def test_exact_symlink_is_materialized_without_download(self):
        target = self.root / "legacy-cache.bin"
        target.write_bytes(b"complete")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)

        with (
            mock.patch.object(install_wizard, "ask") as ask,
            mock.patch.object(install_wizard, "run") as run,
        ):
            self.download({"model.bin": 8})

        self.assertFalse(model.is_symlink())
        self.assertEqual(model.read_bytes(), b"complete")
        ask.assert_not_called()
        run.assert_not_called()

    def test_partial_symlink_is_materialized_before_continuing(self):
        target = self.root / "legacy-partial.bin"
        target.write_bytes(b"part")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)

        def finish_download(_command, **_kwargs):
            self.assertFalse(model.is_symlink())
            self.assertEqual(model.read_bytes(), b"part")
            with model.open("ab") as output:
                output.write(b"ial")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(install_wizard, "ask", return_value=True),
            mock.patch.object(install_wizard, "run", side_effect=finish_download),
        ):
            self.download({"model.bin": 7})

        self.assertEqual(model.read_bytes(), b"partial")
        self.assertEqual(target.read_bytes(), b"part")

    def test_broken_symlink_is_quarantined_before_download(self):
        model = self.model_dir / "model.bin"
        model.symlink_to(self.root / "missing.bin")

        def finish_download(_command, **_kwargs):
            self.assertFalse(model.exists())
            self.assertFalse(model.is_symlink())
            model.write_bytes(b"fresh")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(install_wizard, "ask", return_value=True),
            mock.patch.object(install_wizard, "run", side_effect=finish_download),
        ):
            self.download({"model.bin": 5})

        quarantined = self.model_dir / "model.bin.invalid"
        self.assertTrue(quarantined.is_symlink())
        self.assertEqual(model.read_bytes(), b"fresh")

    def test_oversized_file_is_quarantined_before_download(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"oversized")

        def finish_download(_command, **_kwargs):
            self.assertFalse(model.exists())
            model.write_bytes(b"correct")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(install_wizard, "ask", return_value=True),
            mock.patch.object(install_wizard, "run", side_effect=finish_download),
        ):
            self.download({"model.bin": 7})

        self.assertEqual((self.model_dir / "model.bin.invalid").read_bytes(), b"oversized")
        self.assertEqual(model.read_bytes(), b"correct")

    def test_failed_download_preserves_partial_and_gives_retry_guidance(self):
        model = self.model_dir / "model.bin"
        model.write_bytes(b"part")

        with (
            mock.patch.object(install_wizard, "ask", return_value=True),
            mock.patch.object(
                install_wizard,
                "run",
                return_value=SimpleNamespace(returncode=1),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "重新运行.*续传"):
                self.download({"model.bin": 7})

        self.assertEqual(model.read_bytes(), b"part")

    def test_post_download_size_mismatch_is_rejected(self):
        model = self.model_dir / "model.bin"

        def incomplete_download(_command, **_kwargs):
            model.write_bytes(b"short")
            return SimpleNamespace(returncode=0)

        with (
            mock.patch.object(install_wizard, "ask", return_value=True),
            mock.patch.object(
                install_wizard, "run", side_effect=incomplete_download
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "大小不符"):
                self.download({"model.bin": 7})

    def test_materialization_failure_keeps_symlink_and_cleans_temporary_file(self):
        target = self.root / "legacy-cache.bin"
        target.write_bytes(b"complete")
        model = self.model_dir / "model.bin"
        model.symlink_to(target)

        with mock.patch.object(
            install_wizard.shutil,
            "copyfileobj",
            side_effect=OSError("copy failed"),
        ):
            with self.assertRaisesRegex(OSError, "copy failed"):
                self.download({"model.bin": 8})

        self.assertTrue(model.is_symlink())
        self.assertEqual(list(self.model_dir.glob("model.bin.materializing.*")), [])


class HuggingFaceStageTests(unittest.TestCase):
    def test_pinned_manifests_cover_every_required_model_with_exact_sizes(self):
        self.assertEqual(
            install_wizard.HF_BASE_MANIFEST,
            {
                "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20_970_379_616,
                "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20_970_379_616,
                "text_encoders/H3/qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors": 7_609_128_707,
                "text_encoders/H3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors": 26_363_476_151,
                "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15_687_142_551,
                "upscale_models/RealESRGAN_x2plus.pth": 67_061_725,
                "upscale_models/RealESRGAN_x4plus.pth": 67_040_989,
                "vae/minimax_h3_audio_vae_fp32.safetensors": 605_254_808,
                "vae/minimax_h3_video_vae_fp16.safetensors": 5_207_808_496,
            },
        )
        self.assertEqual(
            install_wizard.HF_EXTRA_MANIFEST,
            {
                "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": 27_141_342_152,
                "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors": 51_506_295_256,
            },
        )
        self.assertRegex(install_wizard.HF_BASE_REVISION, r"^[0-9a-f]{40}$")
        self.assertRegex(install_wizard.HF_EXTRA_REVISION, r"^[0-9a-f]{40}$")

    def test_scratch_stages_use_the_shared_hugging_face_downloader(self):
        with (
            mock.patch.object(install_wizard, "header"),
            mock.patch.object(install_wizard, "scratch_list_models"),
            mock.patch.object(install_wizard, "download_hf_group") as download,
        ):
            install_wizard.scratch_hf_weights()

        download.assert_called_once_with(
            install_wizard.HF_BASE_REPO,
            install_wizard.HF_BASE_REVISION,
            install_wizard.HF_BASE_MANIFEST,
            "Hugging Face 主模型",
        )

        self.assertTrue(
            hasattr(install_wizard, "scratch_hf_extra_text_encoders"),
            "the supplemental text encoders must also use Hugging Face",
        )
        with (
            mock.patch.object(install_wizard, "header"),
            mock.patch.object(install_wizard, "download_hf_group") as download,
        ):
            install_wizard.scratch_hf_extra_text_encoders()

        download.assert_called_once_with(
            install_wizard.HF_EXTRA_REPO,
            install_wizard.HF_EXTRA_REVISION,
            install_wizard.HF_EXTRA_MANIFEST,
            "Hugging Face 补充文本编码器",
        )

    def test_python_environment_has_no_modelscope_dependency(self):
        source = inspect.getsource(install_wizard.step_python_env).lower()
        self.assertNotIn("modelscope", source)


if __name__ == "__main__":
    unittest.main()
