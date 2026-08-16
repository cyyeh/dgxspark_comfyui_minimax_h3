# Hugging Face-only model downloads

## Goal

All fresh-install and clone workflows use Hugging Face as their only remote model source. Re-running either installer must not transfer a model that is already complete, and interrupted downloads must resume without a fixed whole-process timeout.

## Scope

The change covers every tracked runtime and user-facing reference:

- `install_wizard.py`
- `deploy_from_scratch.sh`
- `deploy_to_new_spark.sh`
- `README.md`
- `DEPLOYMENT.md`
- `I2V.md`
- `NEW_SPARK_DEPLOY.md`
- `WORKFLOWS.md`
- `nvidia_forum_post.md`
- automated tests

The model inventory and total download size do not change. Only the source, placement, validation, and resume behavior change.

## Sources and manifests

Stage 4 downloads this fixed manifest from `drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-DGX-Spark-weights`, pinned to revision `2e4f1dbbbc3b42a6c92ac685aa5c4a060c32b05e`:

| Relative path | Expected bytes |
|---|---:|
| `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 |
| `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20,970,379,616 |
| `text_encoders/H3/qwen3vl_32b_h3_generation_tail_50_63_int8_convrot.safetensors` | 7,609,128,707 |
| `text_encoders/H3/qwen3vl_32b_h3_ultra_uncensored_heretic_int8_convrot.safetensors` | 26,363,476,151 |
| `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15,687,142,551 |
| `upscale_models/RealESRGAN_x2plus.pth` | 67,061,725 |
| `upscale_models/RealESRGAN_x4plus.pth` | 67,040,989 |
| `vae/minimax_h3_audio_vae_fp32.safetensors` | 605,254,808 |
| `vae/minimax_h3_video_vae_fp16.safetensors` | 5,207,808,496 |

Stage 5 downloads this fixed manifest from `Comfy-Org/MiniMax-H3`, pinned to revision `d07f69bc8fa09c9717e1e47180034f9322e0e54d`:

| Relative path | Expected bytes |
|---|---:|
| `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27,141,342,152 |
| `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` | 51,506,295,256 |

The manifests are based on the current Hugging Face repository metadata and make installed-file decisions deterministic.

## Installed-file and resume behavior

Before showing a download confirmation or invoking Hugging Face, each stage checks every destination against its exact expected size.

1. A regular file with the expected size is complete and is skipped.
2. A readable symlink whose target is no larger than expected is copied locally and atomically replaces the symlink. An exact-size copy is skipped; an undersized copy becomes the local partial file continued by Wget. This reuses already-installed bytes while removing the old external-cache dependency.
3. A broken or oversized symlink is quarantined before download so Wget can never follow it back into an external cache. Oversized regular files are quarantined for the same reason.
4. A missing or smaller regular file is passed to GNU Wget using its existing destination name and `--continue`.
5. Complete files are never passed to Wget, even when another file in the same stage is missing.
6. A stage with no missing files returns before prompting or starting a subprocess.

Each URL uses the repository's pinned revision and Hugging Face `resolve` endpoint. The Hugging Face CDN advertises byte-range support, and GNU Wget's `--continue` retains a partial destination and requests only the remaining byte range on the next run. Downloads receive no whole-process timeout, while Wget uses a finite retry count so a persistent outage returns clear rerun guidance. A destination larger than its manifest entry is quarantined before retry because it cannot be a valid prefix. After Wget returns, the installer checks the manifest again. Any missing or wrong-size file stops the installation with a precise error and does not create a completion marker.

Completion-marker files are no longer authoritative. Exact final-file validation is performed on every run, so a stale marker cannot hide missing data and an absent marker cannot cause a complete model to download again.

## Fresh-install flows

The interactive installer uses one small manifest-driven helper for both Hugging Face repositories. The non-interactive shell installer embeds the same manifest validation rules and downloads only the returned missing paths. Both installers remove the unused Python package and all cache/symlink setup for the previous remote source. `wget` remains an explicit system dependency.

The existing proxy environment variables remain unchanged and are honored by GNU Wget.

## Clone flows

Model files live under the ComfyUI model directory rather than a separate provider cache. Clone scripts transfer that directory once. `rsync` follows model symlinks so a master created by an older release still produces self-contained regular files on the destination. Separate cache transfer and destination symlink creation are removed.

## Documentation

Every command, stage name, network requirement, source table, resume note, file inventory, clone diagram, and encoder-selection note is changed to Hugging Face terminology and paths. A repository-wide case-insensitive scan must find no stale reference to the removed provider in tracked product files.

All executable installer download commands must fetch from the current `cyyeh/dgxspark_comfyui_minimax_h3` repository through `raw.githubusercontent.com`. No install command, project-home link, or forum-post quick start may point to Gitee or a different GitHub fork. Historical attribution may remain as prose, but cannot be presented as the installation source.

## Error handling

- Network errors leave the partial destination in place and exit with a retry message; Wget continues that file on the next run.
- There is no fixed ten-minute or two-hour timeout around large model downloads.
- A post-download size mismatch lists the exact bad paths.
- A failed symlink materialization leaves the original symlink intact because replacement is atomic.
- Existing complete regular files are never overwritten.

## Tests and verification

Automated tests will cover:

- complete manifest: no prompt and no download subprocess;
- partially complete manifest: Wget receives `--continue`, retains the same partial path, and appends only the simulated remaining bytes;
- retry after the simulated interruption: the completed file is skipped without a second download;
- exact-size symlink: local materialization without network download;
- undersized symlink: local materialization followed by continuation without writing through the link;
- broken or oversized symlink: quarantine before download;
- wrong-size file: selected for repair and rejected if still wrong afterward;
- no fixed timeout on large Hugging Face stages;
- clone commands follow model symlinks and contain no separate cache transfer;
- tracked runtime and user documentation contains no stale provider reference;
- every installer URL and project-home link targets the current GitHub repository, with no executable Gitee source;
- existing nested custom-node regression test remains green.

Final verification includes Python unit tests, Python syntax parsing, shell syntax checks, whitespace checks, a repository-wide stale-reference scan, and inspection of the final Git diff.
