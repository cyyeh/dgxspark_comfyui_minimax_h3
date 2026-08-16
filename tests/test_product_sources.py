import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_FILES = [
    ROOT / "install_wizard.py",
    ROOT / "deploy_from_scratch.sh",
    ROOT / "deploy_to_new_spark.sh",
    ROOT / "README.md",
    ROOT / "DEPLOYMENT.md",
    ROOT / "I2V.md",
    ROOT / "NEW_SPARK_DEPLOY.md",
    ROOT / "WORKFLOWS.md",
    ROOT / "nvidia_forum_post.md",
]
CURRENT_REPO = "github.com/cyyeh/dgxspark_comfyui_minimax_h3"
CURRENT_RAW = "raw.githubusercontent.com/cyyeh/dgxspark_comfyui_minimax_h3/"


class ProductSourceTests(unittest.TestCase):
    def test_product_surface_is_modelscope_free(self):
        for path in PRODUCT_FILES:
            with self.subTest(path=path.name):
                self.assertNotIn("modelscope", path.read_text().lower())

    def test_project_links_use_the_current_github_repository(self):
        for path in PRODUCT_FILES:
            source = path.read_text().lower()
            with self.subTest(path=path.name):
                self.assertNotIn("gitee.com", source)
                self.assertNotIn("luqidaxia/dgxspark_comfyui_minimax_h3", source)
                self.assertNotIn("alexlu0912_admin", source)
                for raw_url in re.findall(r"raw\.githubusercontent\.com/[^\s)`]+", source):
                    self.assertTrue(raw_url.startswith(CURRENT_RAW))

        readme = (ROOT / "README.md").read_text().lower()
        self.assertIn(CURRENT_REPO, readme)

    def test_documented_hugging_face_downloads_are_pinned(self):
        for path in PRODUCT_FILES:
            source = path.read_text().lower()
            with self.subTest(path=path.name):
                self.assertNotRegex(source, r"resolve/(?:main|master)/")
                self.assertNotIn("huggingface-cli download", source)
                self.assertNotIn("huggingface_hub.cli.hf download", source)


if __name__ == "__main__":
    unittest.main()
