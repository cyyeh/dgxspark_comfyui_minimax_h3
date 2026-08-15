import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
