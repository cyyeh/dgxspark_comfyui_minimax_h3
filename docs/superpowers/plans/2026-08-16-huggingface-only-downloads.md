# Hugging Face-only Downloads Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every install and clone path use only pinned Hugging Face model URLs while skipping exact complete files and resuming partial files.

**Architecture:** `install_wizard.py` gets a manifest-driven download unit that prepares legacy symlinks, classifies files by exact expected size, and invokes GNU Wget only for missing or partial destinations. The standalone shell installer mirrors those rules, clone scripts dereference legacy model links, and every tracked document is aligned with the new source and paths.

**Tech Stack:** Python 3.12 standard library, `unittest`, Bash, GNU Wget, rsync, Hugging Face resolve URLs, Git/GitHub.

---

## File map

- `install_wizard.py`: manifests, file preparation, exact-size checks, resumable downloads, interactive stage wiring.
- `deploy_from_scratch.sh`: self-contained equivalent checks and pinned Hugging Face downloads.
- `deploy_to_new_spark.sh`: single self-contained model transfer with symlink dereferencing.
- `tests/test_install_wizard.py`: behavioral tests for installed, partial, invalid, and symlink states.
- `tests/test_huggingface_only.py`: repository-wide and shell/static contract tests.
- `README.md`, `DEPLOYMENT.md`, `I2V.md`, `NEW_SPARK_DEPLOY.md`, `WORKFLOWS.md`, `nvidia_forum_post.md`: user-facing source, command, resume, and clone documentation.

### Task 0: Finalize planning artifacts

**Files:**
- Modify: `docs/superpowers/specs/2026-08-16-huggingface-only-downloads-design.md`
- Create: `docs/superpowers/plans/2026-08-16-huggingface-only-downloads.md`

- [ ] **Step 1: Commit the reviewed spec and plan**

```bash
git add docs/superpowers/specs/2026-08-16-huggingface-only-downloads-design.md docs/superpowers/plans/2026-08-16-huggingface-only-downloads.md
git commit --amend --no-edit
git status -sb
```

Expected: planning artifacts are tracked and the worktree is clean.

### Task 1: Interactive downloader behavior

**Files:**
- Modify: `tests/test_install_wizard.py`
- Modify: `install_wizard.py`

- [ ] **Step 1: Write failing tests for every downloader behavior**

Add tests using tiny manifests rather than real model sizes:

```python
def test_complete_manifest_skips_prompt_and_download(self):
    model = self.model_dir / "text_encoders/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"complete")
    with mock.patch.object(install_wizard, "ask") as ask, \
         mock.patch.object(install_wizard, "run") as run:
        install_wizard.download_hf_group(
            "org/repo", "a" * 40, {"text_encoders/model.bin": 8}, "test"
        )
    ask.assert_not_called()
    run.assert_not_called()

def test_partial_manifest_continues_then_skips_on_retry(self):
    model = self.model_dir / "text_encoders/model.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"part")

    def finish_download(command, **kwargs):
        self.assertIn("--continue", command)
        self.assertIsNone(kwargs["timeout"])
        with model.open("ab") as output:
            output.write(b"ial")
        return SimpleNamespace(returncode=0)

    with mock.patch.object(install_wizard, "ask", return_value=True), \
         mock.patch.object(install_wizard, "run", side_effect=finish_download) as run:
        install_wizard.download_hf_group(
            "org/repo", "b" * 40, {"text_encoders/model.bin": 7}, "test"
        )
        install_wizard.download_hf_group(
            "org/repo", "b" * 40, {"text_encoders/model.bin": 7}, "test"
        )
    self.assertEqual(run.call_count, 1)
```

Before implementation, also add RED tests for exact-size symlink materialization without network access, undersized symlink materialization followed by one continuation, broken symlink quarantine, oversized regular-file quarantine, post-download size mismatch, and a nonzero Wget result that raises retry guidance without deleting the partial file. Add a mixed-manifest case proving a complete file is excluded from Wget while a second partial file continues. Simulate a materialization copy failure and assert the original symlink remains and no `.materializing.*` orphan is left behind.

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_install_wizard -v`

Expected: FAIL because `download_hf_group` is absent.

- [ ] **Step 3: Implement manifest preparation and Wget download**

Add pinned group constants and focused helpers:

```python
import tempfile

HF_BASE_REPO = "drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights"
HF_BASE_REVISION = "2e4f1dbbbc3b42a6c92ac685aa5c4a060c32b05e"
HF_BASE_MANIFEST = {
    "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors": 20_970_379_616,
    "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors": 20_970_379_616,
    "text_encoders/H3/qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors": 7_609_128_707,
    "text_encoders/H3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors": 26_363_476_151,
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": 15_687_142_551,
    "upscale_models/RealESRGAN_x2plus.pth": 67_061_725,
    "upscale_models/RealESRGAN_x4plus.pth": 67_040_989,
    "vae/minimax_h3_audio_vae_fp32.safetensors": 605_254_808,
    "vae/minimax_h3_video_vae_fp16.safetensors": 5_207_808_496,
}
HF_EXTRA_REPO = "Comfy-Org/MiniMax-H3"
HF_EXTRA_REVISION = "d07f69bc8fa09c9717e1e47180034f9322e0e54d"
HF_EXTRA_MANIFEST = {
    "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors": 27_141_342_152,
    "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors": 51_506_295_256,
}

def _quarantine(path):
    candidate = path.with_name(path.name + ".invalid")
    index = 1
    while os.path.lexists(candidate):
        candidate = path.with_name(f"{path.name}.invalid.{index}")
        index += 1
    os.replace(path, candidate)

def _prepare_model_file(path, expected_size):
    if path.is_symlink():
        try:
            target_size = path.stat().st_size
        except OSError:
            _quarantine(path)
            return False
        if target_size > expected_size:
            _quarantine(path)
            return False
        temporary = None
        try:
            fd, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=path.name + ".materializing."
            )
            temporary = Path(temporary_name)
            with os.fdopen(fd, "wb") as output, path.resolve().open("rb") as source:
                shutil.copyfileobj(source, output)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if not path.exists():
        return False
    size = path.stat().st_size
    if size > expected_size:
        _quarantine(path)
        return False
    return size == expected_size

def download_hf_group(repo_id, revision, manifest, label):
    pending = []
    for relative, expected in manifest.items():
        destination = cfg.model_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not _prepare_model_file(destination, expected):
            pending.append((relative, expected))
    if not pending:
        info(f"{label} 已完整安装，跳过")
        return
    if not ask("开始下载？"):
        return
    for relative, expected in pending:
        destination = cfg.model_dir / relative
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{relative}"
        result = run(
            ["wget", "--continue", "--tries=0", "--timeout=60", "--read-timeout=60", url],
            cwd=str(destination.parent), timeout=None, check=False,
        )
        if result is None or result.returncode != 0:
            raise RuntimeError(f"下载中断: {relative}；重新运行安装器将从现有文件续传")
        if not destination.exists() or destination.stat().st_size != expected:
            raise RuntimeError(f"下载后大小不符: {relative}")
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_install_wizard -v`

Expected: all tests pass, including the previous nested custom-node test.

- [ ] **Step 5: Run all downloader regression tests**

Confirm the prewritten complete, mixed-manifest, partial/retry, exact symlink, partial symlink, broken symlink, oversized file, materialization-cleanup, Wget failure, and post-validation tests all pass. Use byte-sized fixtures and assert the destination is a regular file before fake Wget writes.

- [ ] **Step 6: Run tests and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_install_wizard -v`

Commit:

```bash
git add install_wizard.py tests/test_install_wizard.py
git commit -m "Add resumable Hugging Face downloader"
```

### Task 2: Wire both interactive stages

**Files:**
- Modify: `tests/test_install_wizard.py`
- Modify: `install_wizard.py`

- [ ] **Step 1: Write failing stage-wiring tests**

Patch `download_hf_group`, call `scratch_hf_weights()` and the new `scratch_hf_extra_text_encoders()`, and assert each uses the correct repository, pinned revision, and manifest. Add a source assertion that the removed Python package is not installed or imported.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_install_wizard -v`

Expected: FAIL because the extra stage and new wiring are absent.

- [ ] **Step 3: Replace the old cache/symlink stage**

Remove its dependency installation/import, cache flag, download function, and symlink function. Have stage 4 and stage 5 call `download_hf_group`; remove the fixed 7,200-second timeout; rename labels, welcome text, and main-flow call to Hugging Face-only wording.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_install_wizard -v`

- [ ] **Step 5: Commit**

Commit:

```bash
git add install_wizard.py tests/test_install_wizard.py
git commit -m "Use Hugging Face for interactive model stages"
```

### Task 3: Standalone fresh installer

**Files:**
- Create: `tests/test_huggingface_only.py`
- Modify: `deploy_from_scratch.sh`

- [ ] **Step 1: Write failing executable shell tests and manifest contracts**

Make `deploy_from_scratch.sh` sourceable by requiring a standard `BASH_SOURCE` main guard in the finished implementation. In Python tests, create a temporary model directory and a fake `wget` executable, source the shell script, and execute its download functions. Cover: exact complete file never invokes fake Wget; partial regular file is appended to and skipped on the second call; exact and partial symlinks become regular local files; broken/oversized paths are quarantined; failed fake Wget preserves the partial and prints retry guidance; wrong post-download size fails. Use the script's portable Python `os.path.getsize` check, not platform-specific `stat` options, so the tests run on macOS and Linux. Separately assert both pinned revisions and all eleven exact path-to-byte-size pairs.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

- [ ] **Step 3: Implement shell equivalents**

Add `quarantine_path`, `prepare_model_file`, and `download_hf_file` functions plus `if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi`. Query sizes with `${PY:-python3} -c 'import os,sys; print(os.path.getsize(sys.argv[1]))'`, which follows readable links portably. Create materialization paths exclusively with `mktemp` in the destination directory, copy readable symlinks to that regular file, then atomically `mv`. Wrap copy and move failures so the exact temporary file is deleted before returning while the original symlink remains. Invoke Wget from the destination directory without `-O`, validate exact size afterward, and call it for the nine base and two extra manifest entries. Remove cache/symlink and completion-marker logic. Return a nonzero status with explicit rerun/continuation guidance when Wget fails.

- [ ] **Step 4: Validate shell and tests**

Run: `bash -n deploy_from_scratch.sh`

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

- [ ] **Step 5: Commit**

Commit:

```bash
git add deploy_from_scratch.sh tests/test_huggingface_only.py
git commit -m "Use resumable Hugging Face downloads in shell installer"
```

### Task 4: Self-contained clone flows

**Files:**
- Modify: `tests/test_huggingface_only.py`
- Modify: `install_wizard.py`
- Modify: `deploy_to_new_spark.sh`

- [ ] **Step 1: Write failing clone contract tests**

Assert both rsync commands dereference links (`-L` or equivalent), and the standalone clone script contains no second cache transfer or destination symlink-creation stage.

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

- [ ] **Step 3: Simplify clone implementations**

Add `-L` to the interactive rsync options. Remove the standalone cache variable, transfer block, and symlink block; use one `rsync -aL` project transfer so old masters become self-contained destinations.

- [ ] **Step 4: Validate and commit**

Run: `bash -n deploy_to_new_spark.sh`

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

Commit:

```bash
git add install_wizard.py deploy_to_new_spark.sh tests/test_huggingface_only.py
git commit -m "Make cloned model directories self-contained"
```

### Task 5: Documentation consistency

**Files:**
- Modify: `README.md`
- Modify: `DEPLOYMENT.md`
- Modify: `I2V.md`
- Modify: `NEW_SPARK_DEPLOY.md`
- Modify: `WORKFLOWS.md`
- Modify: `nvidia_forum_post.md`
- Modify: `tests/test_huggingface_only.py`

- [ ] **Step 1: Add a failing repository-wide stale-reference test**

Scan only product runtime and user-documentation files (`install_wizard.py`, both deploy shell scripts, `README.md`, `DEPLOYMENT.md`, `I2V.md`, `NEW_SPARK_DEPLOY.md`, `WORKFLOWS.md`, and `nvidia_forum_post.md`) case-insensitively. Fail on the removed provider name, cache paths, CLI commands, or stale stage labels. Also fail when an executable installer URL or project-home link points to Gitee or a GitHub fork other than `cyyeh/dgxspark_comfyui_minimax_h3`. Reject model commands containing `resolve/main`, `resolve/master`, `hf download`, or `huggingface-cli download`; manual model commands must use pinned resolve URLs. Assert the two approved full revision SHAs are used in every manual model command. Tests and historical design/plan documents are deliberately outside this product-surface assertion because they name forbidden values as regression fixtures.

- [ ] **Step 2: Run test and verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

- [ ] **Step 3: Update every user-facing surface**

Change English and Chinese pipeline diagrams, dependency/source tables, resume notes, network requirements, download commands, clone diagrams, total transfer descriptions, reference links, and encoder notes. Point every installer and project-home URL at the current GitHub repository; remove Gitee from executable installation paths. Document exact-size skip and `wget --continue` retry behavior.

- [ ] **Step 4: Run test and verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.test_huggingface_only -v`

- [ ] **Step 5: Commit**

Commit:

```bash
git add README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md tests/test_huggingface_only.py
git commit -m "Document Hugging Face-only deployment"
```

### Task 6: Final verification and publish

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v`

Expected: all tests pass with zero failures/errors.

- [ ] **Step 2: Run syntax and consistency checks**

Run: `python3 -c 'import ast; from pathlib import Path; [ast.parse(p.read_text(), filename=str(p)) for p in [Path("install_wizard.py"), *Path("tests").glob("*.py")]]'`

Run: `bash -n deploy_from_scratch.sh deploy_to_new_spark.sh`

Run: `git diff --check main...HEAD`

Define the exact product paths for the following scans: `install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md`.

Run: `git grep -in 'modelscope' HEAD -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md` and expect exit 1 with no output.

Run: `git grep -in 'gitee.com' HEAD -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md` and expect exit 1 with no output.

Run: `git grep -Ein 'resolve/(main|master)/|(^|[[:space:]])(hf|huggingface-cli)[[:space:]]+download' HEAD -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md` and expect exit 1 with no output.

- [ ] **Step 3: Inspect the final diff and working tree**

Run: `git status -sb && git diff --stat main...HEAD && git diff main...HEAD`

- [ ] **Step 4: Publish through the GitHub workflow**

Push `agent/huggingface-only-downloads`, create a draft PR targeting `main`, confirm it is mergeable and checks are clear, mark it ready, then squash-merge because the user explicitly requested a direct repository update.

- [ ] **Step 5: Verify remote main**

Run these exact commands after the merge:

```bash
git fetch origin main
git switch main
git merge --ff-only origin/main
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
bash -n deploy_from_scratch.sh deploy_to_new_spark.sh
git grep -in 'modelscope' origin/main -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md
git grep -in 'gitee.com' origin/main -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md
git grep -Ein 'resolve/(main|master)/|(^|[[:space:]])(hf|huggingface-cli)[[:space:]]+download' origin/main -- install_wizard.py deploy_from_scratch.sh deploy_to_new_spark.sh README.md DEPLOYMENT.md I2V.md NEW_SPARK_DEPLOY.md WORKFLOWS.md nvidia_forum_post.md
git show origin/main:install_wizard.py | grep -F 'wget' | grep -F -- '--continue'
git show origin/main:install_wizard.py | grep -F '2e4f1dbbbc3b42a6c92ac685aa5c4a060c32b05e'
git show origin/main:install_wizard.py | grep -F 'd07f69bc8fa09c9717e1e47180034f9322e0e54d'
```

The three `git grep` commands must return exit 1 with no matches; every other command must return zero.
