#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║          MiniMax H3 + keys-heretic  引导式安装向导            ║
║              DGX Spark (GB10) 部署                          ║
║                                                              ║
║  支持两种模式:                                                ║
║    A: 从零部署 — 全部从互联网下载                              ║
║    B: 主节点克隆 — 从已有节点高速复制                          ║
║                                                              ║
║  用法: python3 install_wizard.py                             ║
║        python3 install_wizard.py --proxy http://IP:PORT      ║
║        python3 install_wizard.py --yes                       ║
╚══════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import os
import sys
import stat
import time
import shutil
import argparse
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from datetime import timedelta
from urllib.parse import quote

# ── 终端颜色 ─────────────────────────────────────────────────
C = {
    "R": "\033[0m", "B": "\033[1m", "D": "\033[2m",
    "g": "\033[32m", "y": "\033[33m", "r": "\033[31m",
    "c": "\033[36m", "b": "\033[34m", "w": "\033[37m",
    "bg": "\033[1;32m", "br": "\033[1;31m", "bc": "\033[1;36m",
}

def green(s):  return f"{C['g']}{s}{C['R']}"
def yellow(s): return f"{C['y']}{s}{C['R']}"
def red(s):    return f"{C['br']}{s}{C['R']}"
def cyan(s):   return f"{C['bc']}{s}{C['R']}"
def bold(s):   return f"{C['B']}{s}{C['R']}"
def dim(s):    return f"{C['D']}{s}{C['R']}"
def ok():      return green("✓")
def fail():    return red("✗")
def warn():    return yellow("!")
def arrow():   return cyan("→")

# ── 配置 ─────────────────────────────────────────────────────
class Config:
    def __init__(self, args):
        self.mode          = None                    # "scratch" | "clone"
        self.install_dir   = Path(args.install_dir).resolve()
        self.venv_dir      = Path(args.venv_dir).resolve()
        self.comfy_port    = args.port
        self.reserve_vram  = args.reserve_vram
        self.proxy         = args.proxy
        self.yes           = args.yes
        self.master_ip     = args.master_ip or ""
        self.roce_ip       = args.roce_ip or ""
        self.ssh_pass      = args.ssh_pass or ""
        self.comfy_dir     = self.install_dir / "comfy" / "ComfyUI"
        self.model_dir     = self.comfy_dir / "models"
        self.cn_dir        = self.comfy_dir / "custom_nodes"
        self.workflows_dir = self.install_dir / "comfy" / "workflows"
        self.log_dir       = self.install_dir / "logs"
        self.log_file      = self.log_dir / "comfyui.log"
        self.pid_file      = self.log_dir / "comfyui.pid"
        self.python        = self.venv_dir / "bin" / "python3"

        if self.proxy:
            os.environ["http_proxy"] = self.proxy
            os.environ["https_proxy"] = self.proxy
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy

cfg = None


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


def term_width():
    return shutil.get_terminal_size((100, 24)).columns


# ── 进度条 ───────────────────────────────────────────────────
class Bar:
    def __init__(self, total, label="", width=40):
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.start = time.time()
        self._last_len = 0

    def update(self, n):
        pct = min(n / self.total, 1.0)
        filled = int(self.width * pct)
        bar_str = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start
        eta = elapsed / pct - elapsed if pct > 0 and pct < 1 else 0
        eta_str = str(timedelta(seconds=int(eta)))
        line = f"\r  {self.label} [{bar_str}] {pct*100:5.1f}%  ETA {eta_str}"
        pad = self._last_len - len(line)
        sys.stdout.write(line + " " * max(pad, 0))
        sys.stdout.flush()
        self._last_len = len(line)

    def done(self, ok=True):
        elapsed = time.time() - self.start
        status = green("✓") if ok else red("✗")
        sys.stdout.write(f"\r  {self.label} [{status}] {timedelta(seconds=int(elapsed))}\n")
        sys.stdout.flush()


# ── 用户交互 ─────────────────────────────────────────────────
def ask(msg, default=True):
    if cfg.yes:
        return True
    prompt = f"  {arrow()} {msg} {dim('[Y/n]')} " if default else f"  {arrow()} {msg} {dim('[y/N]')} "
    while True:
        try:
            ans = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(); sys.exit(0)
        if ans in ("", "y", "yes"): return True
        if ans in ("n", "no"): return False
        print("  请输入 y 或 n")

def ask_text(msg, default=""):
    prompt = f"  {arrow()} {msg} {dim(f'[{default}]')} " if default else f"  {arrow()} {msg} "
    try:
        return input(prompt).strip() or default
    except (EOFError, KeyboardInterrupt):
        print(); sys.exit(0)

def ask_choice(msg, options):
    """多选一，options = [(key, label)]"""
    print(f"\n  {bold(msg)}")
    for key, label in options:
        print(f"    {cyan(f'[{key}]')}  {label}")
    while True:
        ans = ask_text("请选择", default=options[0][0]).lower()
        for key, _ in options:
            if ans == key.lower():
                return key
        print(f"  请输入: {', '.join(k for k,_ in options)}")

def header(title):
    w = term_width()
    print(f"\n{C['bc']}{'━'*w}{C['R']}")
    print(f"{C['bc']}  [{title}]{C['R']}")
    print(f"{C['bc']}{'━'*w}{C['R']}\n")

def info(msg):    print(f"  {green('•')} {msg}")
def show_cmd(c):  print(f"  {dim('$')} {dim(' '.join(str(x) for x in c))}")


def run(cmd_list, check=True, show=True, cwd=None, capture=False, timeout=600):
    if show:
        show_cmd(cmd_list)
    start = time.time()
    try:
        if capture:
            r = subprocess.run(cmd_list, capture_output=True, text=True,
                               cwd=cwd, timeout=timeout)
        else:
            r = subprocess.run(cmd_list, cwd=cwd, timeout=timeout)
        elapsed = time.time() - start
        if r.returncode != 0:
            print(f"  {fail()} 返回码 {r.returncode} ({timedelta(seconds=int(elapsed))})")
            if check:
                if capture:
                    sys.stderr.write(r.stderr[-500:])
                sys.exit(1)
        else:
            if show or capture:
                print(f"  {ok()} ({timedelta(seconds=int(elapsed))})")
        return r
    except subprocess.TimeoutExpired:
        print(f"  {fail()} 超时")
        if check: sys.exit(1)
        return None
    except KeyboardInterrupt:
        print(f"\n  {warn()} 用户中断"); sys.exit(0)


def _quarantine_model_path(path):
    """Move an unusable model aside without overwriting an earlier quarantine."""
    path = Path(path)
    suffix = ".invalid"
    candidate = path.with_name(path.name + suffix)
    index = 1
    while os.path.lexists(candidate):
        candidate = path.with_name(f"{path.name}{suffix}.{index}")
        index += 1
    os.replace(path, candidate)
    info(f"隔离异常文件: {path.name} → {candidate.name}")
    return candidate


def _prepare_model_path(path, expected_size):
    """Return True only when path is a complete regular file of the expected size."""
    path = Path(path)

    if path.is_symlink():
        try:
            target = path.resolve(strict=True)
            target_size = target.stat().st_size
        except OSError:
            _quarantine_model_path(path)
            return False

        if target_size > expected_size:
            _quarantine_model_path(path)
            return False

        temporary = None
        try:
            fd, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=f"{path.name}.materializing.",
            )
            temporary = Path(temporary_name)
            with os.fdopen(fd, "wb") as destination, target.open("rb") as source:
                shutil.copyfileobj(source, destination)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    if not path.exists():
        return False

    actual_size = path.stat().st_size
    if actual_size > expected_size:
        _quarantine_model_path(path)
        return False
    return actual_size == expected_size


def download_hf_group(repo_id, revision, manifest, label):
    """Download a pinned Hugging Face manifest, skipping or resuming each file."""
    pending = []
    for relative_path, expected_size in manifest.items():
        destination = cfg.model_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _prepare_model_path(destination, expected_size):
            info(f"已安装，跳过: {relative_path}")
        else:
            pending.append((relative_path, expected_size, destination))

    if not pending:
        info(f"{label}: 全部已安装，跳过下载")
        return

    info(f"{label}: 待下载或续传 {len(pending)} 个文件")
    if not ask("开始下载？"):
        return

    for relative_path, expected_size, destination in pending:
        url = (
            f"https://huggingface.co/{repo_id}/resolve/{revision}/"
            f"{quote(relative_path, safe='/')}"
        )
        info(f"下载/续传: {relative_path}")
        result = run(
            [
                "wget",
                "--continue",
                "--tries=10",
                "--timeout=60",
                "--read-timeout=60",
                url,
            ],
            cwd=str(destination.parent),
            timeout=None,
            check=False,
        )
        if result is None or result.returncode != 0:
            raise RuntimeError(
                f"下载中断: {relative_path}；请重新运行安装程序以从现有文件续传"
            )

        actual_size = destination.stat().st_size if destination.exists() else -1
        if actual_size != expected_size:
            raise RuntimeError(
                f"下载完成但大小不符: {relative_path} "
                f"(预期 {expected_size}，实际 {actual_size})；请重新运行以续传"
            )
        info(f"完成: {relative_path} {ok()}")


# ═══════════════════════════════════════════════════════════════
# 公用阶段（两种模式共用）
# ═══════════════════════════════════════════════════════════════

def step_system_deps():
    header("阶段: 系统依赖")
    pkgs = ["git", "ffmpeg", "wget", "ca-certificates"]
    info(f"安装: {', '.join(pkgs)}")
    if not ask("继续？"): return
    run(["apt-get", "update", "-qq"], show=False)
    run(["apt-get", "install", "-y", "-qq"] + pkgs)
    for pkg in ["git", "ffmpeg"]:
        try:
            r = subprocess.run([pkg, "--version"], capture_output=True, text=True)
            v = r.stdout.splitlines()[0] if r.stdout else r.stderr.splitlines()[0]
            info(f"{pkg}: {v}")
        except Exception:
            info(f"{pkg}: {fail()}")

def step_python_env():
    header("阶段: Python 环境")
    py = str(cfg.python)
    info(f"虚拟环境: {cfg.venv_dir}")
    if not cfg.venv_dir.exists():
        run(["python3", "-m", "venv", str(cfg.venv_dir)])
    else:
        info("  已存在，跳过")
    info("PyTorch CUDA 13 (~3 GB)")
    if not ask("继续？"): return
    run([py, "-m", "pip", "install", "-q", "--no-cache-dir",
         "torch", "torchvision", "torchaudio",
         "--index-url", "https://download.pytorch.org/whl/cu130"])
    try:
        r = subprocess.run([py, "-c", "import torch; print(torch.__version__, torch.version.cuda)"],
                           capture_output=True, text=True)
        info(f"PyTorch: {r.stdout.strip()} {ok()}")
    except Exception:
        info(f"PyTorch: {fail()}"); sys.exit(1)
    info("核心依赖")
    run([py, "-m", "pip", "install", "-q", "--no-cache-dir",
         "sageattention==1.0.6", "sqlalchemy", "alembic",
         "pillow", "opencv-python-headless",
         "huggingface_hub"])
    try:
        subprocess.run([py, "-c", "import sageattention, sqlalchemy, huggingface_hub"],
                       capture_output=True, text=True, check=True)
        info(f"核心依赖 {ok()}")
    except Exception:
        info(f"核心依赖 {fail()}"); sys.exit(1)

def step_comfyui_and_nodes():
    header("阶段: ComfyUI + 自定义节点")
    py = str(cfg.python)
    if (cfg.comfy_dir / ".git").exists():
        info(f"ComfyUI 已存在: {cfg.comfy_dir}")
    else:
        info("克隆 ComfyUI v0.30.1...")
        cfg.comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1",
             "https://github.com/comfyanonymous/ComfyUI.git",
             str(cfg.comfy_dir)])
        try:
            subprocess.run(["git", "-C", str(cfg.comfy_dir), "fetch", "--tags",
                           "&&", "git", "-C", str(cfg.comfy_dir), "checkout",
                           "v0.30.1"], shell=True, capture_output=True)
        except Exception: pass

    cfg.cn_dir.mkdir(parents=True, exist_ok=True)
    nodes = [
        ("ComfyUI-SolAttn_triton", "https://github.com/kijai/ComfyUI-SolAttn_triton.git"),
        ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
        ("ComfyUI-Spectrum-MiniMax-H3", "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"),
        ("ComfyUI-H3-Multishot", "https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git"),
        ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ]
    info("克隆自定义节点...")
    for name, url in nodes:
        dst = cfg.cn_dir / name
        if (dst / ".git").exists():
            info(f"  {name} {green('✓')}")
        else:
            run(["git", "clone", "--depth", "1", url, str(dst)], show=False)

    info("Sol-Attn Blackwell + H3 引擎...")
    tmp = Path("/tmp/keys-heretic-tmp")
    if not (tmp / ".git").exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-speed-upgrades-upscaler-finish-Single-DGX-Spark.git",
             str(tmp)], show=False)
    for src_part in ["vendor/ComfyUI_sol-attn_Blackwell"]:
        src = tmp / src_part
        dst = cfg.cn_dir / "ComfyUI_sol-attn_Blackwell"
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    ports_dir = cfg.cn_dir / "h3_sol_engine_ports"
    ports_dir.mkdir(parents=True, exist_ok=True)
    for py_file in ["h3_fbc_node.py", "h3_vae_batch.py"]:
        src = tmp / "nodes" / py_file
        if src.exists():
            shutil.copy2(src, ports_dir / py_file)
            dst2 = cfg.cn_dir / "ComfyUI_sol-attn_Blackwell" / py_file
            shutil.copy2(src, dst2)
    init_py = ports_dir / "__init__.py"
    init_py.write_text("""from .h3_fbc_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
try:
    from .h3_vae_batch import install as _install_vae_batch
    _install_vae_batch()
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("H3 batched VAE not installed: %s", e)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
""")
    info(f"  Sol-Attn {ok()}")
    run([py, "-m", "pip", "install", "-q", "--no-cache-dir",
         "-r", str(cfg.comfy_dir / "requirements.txt")])
    cfg.workflows_dir.mkdir(parents=True, exist_ok=True)
    wf_src = tmp / "workflows"
    if wf_src.exists():
        for f in wf_src.glob("*.json"):
            shutil.copy2(f, cfg.workflows_dir / f.name)
    n_wf = len(list(cfg.workflows_dir.glob("*.json")))
    info(f"工作流: {n_wf} 个 {ok()}")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  {ok()} ComfyUI + 节点 完成\n")

def step_start_comfyui():
    header("阶段: 启动 ComfyUI")
    py = str(cfg.python)
    subprocess.run(["pkill", "-f", f"main.py.*{cfg.comfy_port}"], capture_output=True)
    time.sleep(1)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    info(f"启动 (端口 {cfg.comfy_port})...")
    with open(cfg.log_file, "w") as log:
        subprocess.Popen(
            [py, "main.py", "--listen", "0.0.0.0",
             "--port", str(cfg.comfy_port),
             "--reserve-vram", str(cfg.reserve_vram)],
            cwd=str(cfg.comfy_dir), stdout=log, stderr=subprocess.STDOUT, env=env)
    time.sleep(2)
    pid = subprocess.run(["pgrep", "-f", f"main.py.*{cfg.comfy_port}"],
                         capture_output=True, text=True).stdout.strip()
    if pid:
        cfg.pid_file.write_text(pid); info(f"PID: {pid}")
    else:
        info(f"PID: 未获取 {warn()}")
    info("等待就绪...")
    bar = Bar(45, "启动中", 30)
    started = False
    for i in range(1, 46):
        bar.update(i); time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://localhost:{cfg.comfy_port}/", timeout=2)
            if r.status == 200:
                bar.done(True); started = True; break
        except Exception: pass
    if not started:
        bar.done(False)
        print(f"  {warn()} 可能仍在加载模型: tail -f {cfg.log_file}")
    write_scripts(py)

def write_scripts(py):
    restart = cfg.install_dir / "restart.sh"
    restart.write_text(f"""#!/bin/bash
kill $(cat {cfg.pid_file} 2>/dev/null) 2>/dev/null
sleep 2
cd {cfg.comfy_dir}
nohup {py} main.py --listen 0.0.0.0 --port {cfg.comfy_port} --reserve-vram {cfg.reserve_vram} \\
    > {cfg.log_file} 2>&1 &
echo $! > {cfg.pid_file}
echo "ComfyUI restarted, PID: $(cat {cfg.pid_file})"
""")
    restart.chmod(restart.stat().st_mode | stat.S_IEXEC)
    status = cfg.install_dir / "status.sh"
    status.write_text(f"""#!/bin/bash
echo "PID:    $(cat {cfg.pid_file} 2>/dev/null || echo 'N/A')"
echo "HTTP:   $(curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{cfg.comfy_port}/ 2>/dev/null || echo 'DOWN')"
echo "磁盘:   $(df -h / | awk 'NR==2{{print $3" / "$2" ("$5" used)"}}')"
echo "模型:   $(find {cfg.model_dir} -name '*.safetensors' -o -name '*.pth' 2>/dev/null | wc -l) 个"
""")
    status.chmod(status.stat().st_mode | stat.S_IEXEC)

def step_finish():
    try:
        ip = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        ip = ip.stdout.strip().split()[0]
    except Exception: ip = "localhost"
    w = term_width()
    print(f"\n{C['bg']}{'='*w}{C['R']}")
    print(f"{C['bg']}  🎉 部署完成!{C['R']}")
    print(f"{C['bg']}{'='*w}{C['R']}\n")
    print(f"  {bold('ComfyUI')}    {cyan(f'http://{ip}:{cfg.comfy_port}')}")
    print(f"  {bold('安装目录')}   {cfg.install_dir}")
    print(f"  {bold('Python')}     {cfg.python}\n")
    print(f"  {bold('管理命令:')}")
    print(f"    bash {cfg.install_dir}/restart.sh")
    print(f"    bash {cfg.install_dir}/status.sh")
    print(f"    tail -f {cfg.log_file}\n")
    print(f"  {bold('快速使用:')}")
    print(f"    1. 浏览器打开 {cyan(f'http://{ip}:{cfg.comfy_port}')}")
    print(f"    2. 拖入工作流: {cfg.workflows_dir}")
    print(f"    3. 推荐从 h3-dense-baseline.json 开始")
    print(f"    4. 修改 prompt → Queue Prompt")
    print()


# ═══════════════════════════════════════════════════════════════
# 模式 A: 从零部署（全从互联网下载）
# ═══════════════════════════════════════════════════════════════

def scratch_hf_weights():
    header("下载权重 — Hugging Face 主模型 (~91 GB)")
    download_hf_group(
        HF_BASE_REPO,
        HF_BASE_REVISION,
        HF_BASE_MANIFEST,
        "Hugging Face 主模型",
    )
    scratch_list_models()

def scratch_list_models():
    models = list(cfg.model_dir.rglob("*.safetensors")) + list(cfg.model_dir.rglob("*.pth"))
    if models:
        total = sum(f.stat().st_size for f in models)
        info(f"已下载 {len(models)} 个，共 {total/(1024**3):.1f} GB")
        for m in sorted(models):
            rel = m.relative_to(cfg.model_dir)
            print(f"    {dim(f'{m.stat().st_size/(1024**3):5.1f} GB')}  {rel}")

def scratch_hf_extra_text_encoders():
    header("下载权重 — Hugging Face 补充文本编码器 (~74 GB)")
    download_hf_group(
        HF_EXTRA_REPO,
        HF_EXTRA_REVISION,
        HF_EXTRA_MANIFEST,
        "Hugging Face 补充文本编码器",
    )


# ═══════════════════════════════════════════════════════════════
# 模式 B: 主节点克隆（从已有节点高速复制）
# ═══════════════════════════════════════════════════════════════

def clone_detect_master():
    header("检测主节点")
    if cfg.master_ip:
        info(f"主节点管理IP: {cfg.master_ip}")
    else:
        cfg.master_ip = ask_text("主节点管理 IP", "192.168.22.xxx")
    if cfg.roce_ip:
        info(f"主节点 RoCE IP: {cfg.roce_ip}")
    else:
        cfg.roce_ip = ask_text("主节点 RoCE IP (高速内网)", "10.10.12.xxx")
    if cfg.ssh_pass:
        info("SSH 密码已设置")
    else:
        cfg.ssh_pass = ask_text("主节点 SSH root 密码", "")
    if not cfg.ssh_pass:
        print(f"  {warn()} 未设置密码，将尝试密钥认证\n")
    info("测试 SSH 连通性...")
    ssh_cmd = ["sshpass", "-p", cfg.ssh_pass, "ssh",
               "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
               f"root@{cfg.master_ip}", "echo OK"] if cfg.ssh_pass else \
              ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
               f"root@{cfg.master_ip}", "echo OK"]
    r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        info(f"SSH {cfg.master_ip} {ok()}")
    else:
        info(f"SSH {fail()} {r.stderr.strip()[-120:]}")
        print(f"  {dim('请检查 IP 和密码后重试，或选择方案 A')}")
        sys.exit(1)
    # 确认源路径
    r = subprocess.run(
        ssh_cmd[:-1] + ["test -d /root/minnimax-h3/comfy/ComfyUI/models && echo EXISTS || echo MISSING"],
        capture_output=True, text=True, timeout=10)
    if "EXISTS" in r.stdout:
        info(f"主节点 MiniMax H3 已部署 {ok()}")
    else:
        info(f"主节点未找到 /root/minnimax-h3/comfy/ComfyUI/models {warn()}")
        if not ask("继续尝试复制？"):
            sys.exit(1)

def clone_rsync_models():
    header("高速复制模型和数据 (~145 GB)")
    src = f"root@{cfg.roce_ip}:/root/minnimax-h3/comfy/ComfyUI/models/"
    # 试 RoCE 直连
    info(f"尝试 RoCE 直连: {cfg.roce_ip}...")
    r = subprocess.run(
        ["sshpass", "-p", cfg.ssh_pass, "ssh",
         "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         f"root@{cfg.roce_ip}", "echo OK"] if cfg.ssh_pass else
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
         f"root@{cfg.roce_ip}", "echo OK"],
        capture_output=True, text=True, timeout=10)
    rsync_ip = cfg.roce_ip if r.returncode == 0 else cfg.master_ip
    if rsync_ip == cfg.roce_ip:
        info(f"使用 RoCE 高速网络 {ok()}")
    else:
        info(f"RoCE 不可达，回退到管理网络 {cfg.master_ip}")

    print(f"\n  {bold('即将 rsync:')}")
    print(f"    {dim(rsync_ip)}:{src} → {cfg.model_dir}")
    print(f"  {dim('约 145GB，RoCE 下 ~5 分钟，管理网络 ~30 分钟')}\n")
    if not ask("开始传输？"):
        print(f"\n  {dim('跳过。')}\n"); return

    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    info("rsync 传输中...")
    env = os.environ.copy()
    if cfg.ssh_pass:
        env["RSYNC_PASSWORD"] = cfg.ssh_pass
    ssh_opt = f"ssh -o StrictHostKeyChecking=no"
    if cfg.ssh_pass:
        ssh_opt = f"sshpass -p {cfg.ssh_pass} ssh -o StrictHostKeyChecking=no"
    cmd = ["rsync", "-aLz", "--progress", "-e", ssh_opt,
           f"root@{rsync_ip}:/root/minnimax-h3/comfy/ComfyUI/models/",
           str(cfg.model_dir) + "/"]
    show_cmd(cmd)
    start = time.time()
    r = subprocess.run(cmd, timeout=3600)
    if r.returncode == 0:
        info(f"rsync 完成 ({timedelta(seconds=int(time.time()-start))}) {ok()}")
    else:
        info(f"rsync 异常 {warn()}，请检查")

    # 也复制工作流
    info("同步工作流...")
    wf_src = f"root@{rsync_ip}:/root/minnimax-h3/comfy/workflows/"
    cfg.workflows_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-avz", "-e", ssh_opt, wf_src, str(cfg.workflows_dir) + "/"],
        timeout=300)
    n_wf = len(list(cfg.workflows_dir.glob("*.json")))
    info(f"工作流: {n_wf} 个 {ok()}")


# ═══════════════════════════════════════════════════════════════
# 前置检查
# ═══════════════════════════════════════════════════════════════

def step_check():
    header("前置检查")
    issues = []
    if os.geteuid() != 0:
        issues.append("需要 root 权限")
    try:
        r = subprocess.run(["python3", "--version"], capture_output=True, text=True)
        info(f"Python: {r.stdout.strip() or r.stderr.strip()} {ok()}")
    except FileNotFoundError:
        issues.append("未找到 python3")
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        info(f"GPU:    已检测 {ok()}")
    except Exception:
        info(f"GPU:    {yellow('未检测')}")
    try:
        disk = os.statvfs(cfg.install_dir if cfg.install_dir.exists() else "/")
        free_gb = disk.f_frsize * disk.f_bavail / (1024**3)
        if free_gb < 200:
            issues.append(f"磁盘仅 {free_gb:.0f} GB，建议 ≥200 GB")
            info(f"磁盘:   {free_gb:.0f} GB 可用 {warn()}")
        else:
            info(f"磁盘:   {free_gb:.0f} GB 可用 {ok()}")
    except Exception:
        info(f"磁盘:   无法检测 {warn()}")
    try:
        urllib.request.urlopen("https://github.com", timeout=5)
        info(f"GitHub: 可达 {ok()}")
    except Exception:
        info(f"GitHub: {yellow('不可达')}")
        if not cfg.proxy:
            p = ask_text("设置 HTTP 代理？(留空跳过)", "")
            if p:
                cfg.proxy = p
                os.environ["http_proxy"] = p; os.environ["https_proxy"] = p
                os.environ["HTTP_PROXY"] = p; os.environ["HTTPS_PROXY"] = p
                info(f"代理已设: {p}")
    if issues:
        print(f"\n  {fail()} 问题:")
        for i in issues: print(f"     • {i}")
        if not ask("忽略继续？", default=False): sys.exit(1)
    else:
        print(f"\n  {ok()} 检查通过\n")


# ═══════════════════════════════════════════════════════════════
# 欢迎页 & 模式选择
# ═══════════════════════════════════════════════════════════════

def step_welcome():
    w = term_width()
    print(f"\n{C['bg']}{'='*w}{C['R']}")
    print(f"{C['bg']}  MiniMax H3 + keys-heretic  引导式安装向导{C['R']}")
    print(f"{C['bg']}  DGX Spark (GB10) 部署{C['R']}")
    print(f"{C['bg']}{'='*w}{C['R']}\n")
    print(f"  {bold('安装目录:')}   {cfg.install_dir}")
    print(f"  {bold('虚拟环境:')}   {cfg.venv_dir}")
    print(f"  {bold('端口:')}      {cfg.comfy_port}")
    print(f"  {bold('代理:')}      {cfg.proxy or '无'}")
    print()

    # ── 模式选择 ──
    if cfg.master_ip:
        mode = "clone"
        info(f"命令行指定主节点 {cfg.master_ip}，自动选择 {bold('方案 B: 主节点克隆')}")
    else:
        mode = ask_choice("选择部署方案:", [
            ("A", "从零部署 — 全部从 Hugging Face 下载 (~165 GB)"),
            ("B", "主节点克隆 — 从已部署的 DGX Spark 高速复制 (~5 分钟)"),
        ])
    if mode.lower() == "a":
        cfg.mode = "scratch"
    else:
        cfg.mode = "clone"

    print(f"\n  {bold('方案:')} ", end="")
    if cfg.mode == "scratch":
        print(f"{cyan('A — 从零部署 (互联网下载 ~165 GB)')}")
    else:
        print(f"{cyan('B — 主节点克隆 (高速复制 ~145 GB)')}")
    print()

    if not cfg.yes:
        print(f"  {dim('提示: 加 --yes 跳过所有确认')}\n")
    if not ask("开始安装？"):
        print(f"\n  {dim('已取消。')}\n"); sys.exit(0)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    global cfg
    parser = argparse.ArgumentParser(description="MiniMax H3 引导式安装向导")
    parser.add_argument("--install-dir", default="/root/minnimax-h3")
    parser.add_argument("--venv-dir", default="/opt/minnimax-h3-venv")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--reserve-vram", type=int, default=8)
    parser.add_argument("--proxy", default="")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="跳过所有确认提示")
    parser.add_argument("--master-ip", default="",
                        help="主节点管理 IP (方案B自动选择)")
    parser.add_argument("--roce-ip", default="",
                        help="主节点 RoCE IP (方案B)")
    parser.add_argument("--ssh-pass", default="",
                        help="主节点 SSH 密码 (方案B)")
    args = parser.parse_args()
    cfg = Config(args)
    cfg.install_dir.mkdir(parents=True, exist_ok=True)

    try:
        step_welcome()
        step_check()

        # ── 两种模式分流 ──
        if cfg.mode == "clone":
            clone_detect_master()
            step_system_deps()          # 共用
            step_python_env()           # 共用
            step_comfyui_and_nodes()    # 共用
            clone_rsync_models()        # B 专属：rsync 替代下载
            step_start_comfyui()        # 共用
        else:
            step_system_deps()
            step_python_env()
            step_comfyui_and_nodes()
            scratch_hf_weights()        # A 专属
            scratch_hf_extra_text_encoders()  # A 专属
            step_start_comfyui()

        step_finish()
    except KeyboardInterrupt:
        print(f"\n\n  {warn()} 安装中断"); sys.exit(0)


if __name__ == "__main__":
    main()
