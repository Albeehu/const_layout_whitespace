#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#改完S_tb 跑推論用
"""
Run official inference for multiple whitespace styles in one command.

This script is a wrapper around infer_official_face_v5_frame_summary.py.
It runs each style one by one, creates an output folder for every style,
and saves a terminal log for every run.

Example:
python run_multi_whitespace_infer.py \
  --infer_script /home/albee/const_layout_whitespace/infer_official_face_v5_frame_summary.py \
  --base_out_dir /home/albee/const_layout_whitespace/final_eval/official_infer/multi_face \
  --ckpt_dir /home/albee/const_layout_whitespace/final_eval/ckpts \
  --styles frame right top hybrid \
  --n 200 \
  --k 64 \
  --seed 123 \
  --max_fg_area 0.33 \
  --max_elems 5 \
  --add_face 1 \
  --svg_small_prob 0.72

Expected default ckpt names under --ckpt_dir:
  frame  -> frame_final.pth
  right  -> right_final.pth
  top    -> top_final.pth
  hybrid -> hybrid_final.pth

You can override any checkpoint path with:
  --frame_ckpt  /path/to/frame.pth
  --right_ckpt  /path/to/right.pth
  --top_ckpt    /path/to/top.pth
  --hybrid_ckpt /path/to/hybrid.pth
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


VALID_STYLES = ["frame", "right", "top", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-style whitespace inference")

    parser.add_argument(
        "--infer_script",
        type=str,
        default="/home/albee/const_layout_whitespace/infer_official_face_v5_frame_summary.py",
        help="Path to the single-style inference script.",
    )
    parser.add_argument(
        "--base_out_dir",
        type=str,
        required=True,
        help="Base output directory. Each style will be saved into base_out_dir/<style>_face.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/home/albee/const_layout_whitespace/final_eval/ckpts",
        help="Directory containing <style>_final.pth checkpoints.",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=VALID_STYLES,
        choices=VALID_STYLES,
        help="Styles to run.",
    )

    # Optional checkpoint overrides
    parser.add_argument("--frame_ckpt", type=str, default="")
    parser.add_argument("--right_ckpt", type=str, default="")
    parser.add_argument("--top_ckpt", type=str, default="")
    parser.add_argument("--hybrid_ckpt", type=str, default="")

    # Shared inference parameters
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_fg_area", type=float, default=0.33)
    parser.add_argument("--max_elems", type=int, default=5)
    parser.add_argument("--add_face", type=int, default=1)
    parser.add_argument("--svg_small_prob", type=float, default=0.72)

    # Frame-specific parameters. They are passed to all styles, but only frame uses them.
    parser.add_argument("--frame_band_margin", type=float, default=0.10)
    parser.add_argument("--frame_target_w", type=float, default=0.28)
    parser.add_argument("--frame_target_h", type=float, default=0.38)
    parser.add_argument("--frame_max_ar", type=float, default=1.8)

    # Optional baseline summaries for comparison
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default="",
        help="Optional base directory containing baseline folders, e.g. baseline_dir/frame_baseline/summary_stats.json.",
    )
    parser.add_argument(
        "--compare_suffix",
        type=str,
        default="_baseline",
        help="Folder suffix for baseline comparison. Default uses <style>_baseline.",
    )

    # Logging / execution
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)

    return parser.parse_args()


def resolve_ckpts(args: argparse.Namespace) -> Dict[str, Path]:
    ckpt_dir = Path(args.ckpt_dir)
    overrides = {
        "frame": args.frame_ckpt,
        "right": args.right_ckpt,
        "top": args.top_ckpt,
        "hybrid": args.hybrid_ckpt,
    }

    ckpts: Dict[str, Path] = {}
    for style in VALID_STYLES:
        if overrides[style]:
            ckpts[style] = Path(overrides[style])
        else:
            ckpts[style] = ckpt_dir / f"{style}_final.pth"
    return ckpts


def build_command(args: argparse.Namespace, style: str, ckpt_path: Path, out_dir: Path) -> List[str]:
    cmd = [
        args.python_bin,
        args.infer_script,
        "--resume_ckpt", str(ckpt_path),
        "--out_dir", str(out_dir),
        "--style", style,
        "--n", str(args.n),
        "--k", str(args.k),
        "--seed", str(args.seed),
        "--max_fg_area", str(args.max_fg_area),
        "--max_elems", str(args.max_elems),
        "--add_face", str(args.add_face),
        "--svg_small_prob", str(args.svg_small_prob),
        "--frame_band_margin", str(args.frame_band_margin),
        "--frame_target_w", str(args.frame_target_w),
        "--frame_target_h", str(args.frame_target_h),
        "--frame_max_ar", str(args.frame_max_ar),
        "--summary_name", f"improved_{style}",
    ]

    if args.baseline_dir:
        compare_json = Path(args.baseline_dir) / f"{style}{args.compare_suffix}" / "summary_stats.json"
        if compare_json.exists():
            cmd += [
                "--compare_stats_json", str(compare_json),
                "--compare_name", f"baseline_{style}",
            ]
        else:
            print(f"[WARN] baseline summary not found for {style}: {compare_json}")

    return cmd


def run_and_log(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n[RUN]", " ".join(cmd))
    print("[LOG]", log_path)

    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        return proc.wait()


def write_master_summary(base_out_dir: Path, styles: List[str]) -> None:
    """Collect each style's summary.txt into one easy-to-open text file."""
    master_path = base_out_dir / "all_styles_summary.txt"
    parts = []
    for style in styles:
        summary_path = base_out_dir / f"{style}_face" / "summary.txt"
        if summary_path.exists():
            parts.append(f"########## {style.upper()} ##########\n")
            parts.append(summary_path.read_text(encoding="utf-8"))
            parts.append("\n")
        else:
            parts.append(f"########## {style.upper()} ##########\nsummary.txt not found: {summary_path}\n\n")
    master_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\n[INFO] master summary saved: {master_path}")


def main() -> None:
    args = parse_args()

    infer_script = Path(args.infer_script)
    if not infer_script.exists():
        raise FileNotFoundError(f"infer_script not found: {infer_script}")

    base_out_dir = Path(args.base_out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpts = resolve_ckpts(args)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    failed = []

    for style in args.styles:
        ckpt_path = ckpts[style]
        if not ckpt_path.exists():
            print(f"[ERROR] checkpoint not found for {style}: {ckpt_path}")
            failed.append(style)
            continue

        out_dir = base_out_dir / f"{style}_face"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(args, style, ckpt_path, out_dir)
        log_path = log_dir / f"infer_{style}_{timestamp}.log"

        if args.dry_run:
            print("\n[DRY RUN]", " ".join(cmd))
            print("[DRY LOG]", log_path)
            continue

        code = run_and_log(cmd, log_path)
        if code != 0:
            print(f"[ERROR] style={style} failed with exit code {code}")
            failed.append(style)
        else:
            print(f"[DONE] style={style} saved to {out_dir}")

    if not args.dry_run:
        write_master_summary(base_out_dir, args.styles)

    if failed:
        print("\n[FAILED]", ", ".join(failed))
        sys.exit(1)

    print("\n[ALL DONE]")


if __name__ == "__main__":
    main()
