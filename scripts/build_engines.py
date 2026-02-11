#!/usr/bin/env python3
"""
build_engines.py - Build TensorRT engines from ONNX models.

Uses trtexec subprocess to convert ONNX models to TensorRT .trt engines
with optimized dynamic shape profiles.

Usage:
    python scripts/build_engines.py --onnx-dir onnx/<variant> --output-dir engines/<variant> [--fp16]

Examples:
    python scripts/build_engines.py --onnx-dir onnx/tts_1.5b --output-dir engines/tts_1.5b --fp16
    python scripts/build_engines.py --onnx-dir onnx/tts_0.5b --output-dir engines/tts_0.5b --fp16
    python scripts/build_engines.py --onnx-dir onnx/asr --output-dir engines/asr --fp16
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def find_trtexec():
    """Find trtexec executable."""
    # Check TensorRT root
    trt_root = os.environ.get("TENSORRT_ROOT", "")
    if trt_root:
        candidates = [
            os.path.join(trt_root, "bin", "trtexec.exe"),
            os.path.join(trt_root, "bin", "trtexec"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c

    # Check PATH
    trtexec = shutil.which("trtexec") or shutil.which("trtexec.exe")
    if trtexec:
        return trtexec

    raise FileNotFoundError(
        "trtexec not found. Set TENSORRT_ROOT environment variable or add trtexec to PATH."
    )


def build_engine(trtexec_path, onnx_path, engine_path, shapes=None, fp16=False,
                 extra_args=None):
    """Build a TensorRT engine from an ONNX model using trtexec.

    Args:
        trtexec_path: Path to trtexec executable
        onnx_path: Path to input ONNX model
        engine_path: Path to output .trt engine
        shapes: Dict of {profile_name: (min_shapes, opt_shapes, max_shapes)}
                Each shapes value is a string like "input_name:1x1x24000"
        fp16: Enable FP16 precision
        extra_args: Additional trtexec arguments
    """
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
    ]

    if fp16:
        cmd.append("--fp16")

    if shapes:
        min_parts = []
        opt_parts = []
        max_parts = []
        for name, (min_s, opt_s, max_s) in shapes.items():
            min_parts.append(f"{name}:{min_s}")
            opt_parts.append(f"{name}:{opt_s}")
            max_parts.append(f"{name}:{max_s}")

        cmd.append(f"--minShapes={','.join(min_parts)}")
        cmd.append(f"--optShapes={','.join(opt_parts)}")
        cmd.append(f"--maxShapes={','.join(max_parts)}")

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n  Running: {' '.join(cmd)}")
    print(f"  (This may take several minutes...)")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"  STDERR:\n{result.stderr}")
        print(f"  STDOUT:\n{result.stdout}")
        raise RuntimeError(f"trtexec failed with return code {result.returncode}")

    if os.path.isfile(engine_path):
        size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  OK: {engine_path} ({size_mb:.1f} MB)")
    else:
        raise FileNotFoundError(f"Engine file not created: {engine_path}")


def get_shape_profiles(model_type: str, metadata: dict):
    """Get dynamic shape profiles for each sub-model.

    Returns dict of {onnx_name: {input_name: (min, opt, max)}}
    """
    H = metadata["hidden_size"]
    num_layers = metadata["num_hidden_layers"]
    num_kv_heads = metadata["num_key_value_heads"]
    head_dim = metadata["head_dim"]
    V = metadata["vocab_size"]

    profiles = {}

    # Acoustic encoder: audio [B, 1, N]
    profiles["acoustic_encoder"] = {
        "audio": ("1x1x24000", "1x1x240000", "1x1x14400000"),
    }

    # Acoustic decoder: latent [B, 64, T]
    if model_type != "asr":
        profiles["acoustic_decoder"] = {
            "latent": ("1x64x1", "1x64x100", "1x64x4500"),
        }

    # Semantic encoder: audio [B, 1, N]
    if model_type != "tts_0.5b":
        profiles["semantic_encoder"] = {
            "audio": ("1x1x24000", "1x1x240000", "1x1x14400000"),
        }

    # Diffusion head: noisy [B, 64], timesteps [B], condition [B, H]
    if model_type != "asr":
        profiles["diffusion_head"] = {
            "noisy_images": ("1x64", "2x64", "4x64"),
            "timesteps": ("1", "2", "4"),
            "condition": (f"1x{H}", f"2x{H}", f"4x{H}"),
        }

    # Language model profiles
    if model_type == "tts_0.5b":
        base_layers = metadata.get("base_lm_layers", 4)
        tts_layers = metadata.get("tts_lm_layers", 20)

        # base_lm prefill
        base_prefill_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x512x{H}", f"1x4096x{H}"),
            "position_ids": ("1x1", "1x512", "1x4096"),
        }
        profiles["base_lm_prefill"] = base_prefill_shapes

        # base_lm decode
        base_decode_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x1x{H}", f"1x1x{H}"),
            "position_ids": ("1x1", "1x1", "1x1"),
        }
        for i in range(base_layers):
            base_decode_shapes[f"past_key_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x4096x{head_dim}",
            )
            base_decode_shapes[f"past_value_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x4096x{head_dim}",
            )
        profiles["base_lm_decode"] = base_decode_shapes

        # tts_lm prefill
        tts_prefill_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x512x{H}", f"1x4096x{H}"),
            "position_ids": ("1x1", "1x512", "1x4096"),
        }
        profiles["tts_lm_prefill"] = tts_prefill_shapes

        # tts_lm decode
        tts_decode_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x1x{H}", f"1x1x{H}"),
            "position_ids": ("1x1", "1x1", "1x1"),
        }
        for i in range(tts_layers):
            tts_decode_shapes[f"past_key_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x4096x{head_dim}",
            )
            tts_decode_shapes[f"past_value_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x4096x{head_dim}",
            )
        profiles["tts_lm_decode"] = tts_decode_shapes

    else:
        # Full LM (1.5B or ASR)
        max_seq = 4096
        if model_type == "asr":
            max_seq = 32768  # ASR needs longer sequences

        # prefill
        prefill_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x512x{H}", f"1x{max_seq}x{H}"),
            "position_ids": ("1x1", "1x512", f"1x{max_seq}"),
            "attention_mask": ("1x1x1x1", "1x1x512x512", f"1x1x{max_seq}x{max_seq}"),
        }
        profiles["language_model_prefill"] = prefill_shapes

        # decode
        decode_shapes = {
            "inputs_embeds": (f"1x1x{H}", f"1x1x{H}", f"1x1x{H}"),
            "position_ids": ("1x1", "1x1", "1x1"),
            "attention_mask": ("1x1x1x2", f"1x1x1x513", f"1x1x1x{max_seq + 1}"),
        }
        for i in range(num_layers):
            decode_shapes[f"past_key_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x{max_seq}x{head_dim}",
            )
            decode_shapes[f"past_value_{i}"] = (
                f"1x{num_kv_heads}x1x{head_dim}",
                f"1x{num_kv_heads}x512x{head_dim}",
                f"1x{num_kv_heads}x{max_seq}x{head_dim}",
            )
        profiles["language_model_decode"] = decode_shapes

    return profiles


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines from ONNX models")
    parser.add_argument(
        "--onnx-dir", type=str, required=True,
        help="Directory containing ONNX files (output of export_onnx.py)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for TensorRT engine files",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Enable FP16 precision (default: True)",
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable FP16 precision",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Only build engine for this specific model (e.g., 'acoustic_encoder')",
    )
    args = parser.parse_args()

    fp16 = args.fp16 and not args.no_fp16

    # Find trtexec
    trtexec = find_trtexec()
    print(f"Using trtexec: {trtexec}")

    # Load metadata
    metadata_path = os.path.join(args.onnx_dir, "model_metadata.json")
    if not os.path.isfile(metadata_path):
        print(f"ERROR: model_metadata.json not found at {metadata_path}")
        print("Run export_onnx.py first.")
        sys.exit(1)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model_type = metadata["model_type"]
    print(f"Model type: {model_type}")
    print(f"FP16: {fp16}")

    # Get shape profiles
    profiles = get_shape_profiles(model_type, metadata)

    os.makedirs(args.output_dir, exist_ok=True)

    # Build engines
    built = 0
    failed = 0

    for onnx_name, shapes in profiles.items():
        if args.only and args.only != onnx_name:
            continue

        onnx_path = os.path.join(args.onnx_dir, f"{onnx_name}.onnx")
        engine_path = os.path.join(args.output_dir, f"{onnx_name}.trt")

        if not os.path.isfile(onnx_path):
            print(f"\n  WARNING: {onnx_path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Building engine: {onnx_name}")
        print(f"  ONNX: {onnx_path}")
        print(f"  Engine: {engine_path}")
        print(f"{'='*60}")

        try:
            build_engine(trtexec, onnx_path, engine_path, shapes=shapes, fp16=fp16)
            built += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    # Copy weights directory
    weights_src = os.path.join(args.onnx_dir, "weights")
    weights_dst = os.path.join(args.output_dir, "weights")
    if os.path.isdir(weights_src):
        print(f"\nCopying weights directory...")
        if os.path.isdir(weights_dst):
            shutil.rmtree(weights_dst)
        shutil.copytree(weights_src, weights_dst)
        weight_files = os.listdir(weights_dst)
        print(f"  Copied {len(weight_files)} weight files")

    # Copy metadata
    metadata_dst = os.path.join(args.output_dir, "model_metadata.json")
    shutil.copy2(metadata_path, metadata_dst)

    # Summary
    print(f"\n{'='*60}")
    print(f"Build complete: {model_type}")
    print(f"  Built: {built} engines")
    print(f"  Failed: {failed} engines")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    for root, dirs, files in os.walk(args.output_dir):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, args.output_dir)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"  {rel}  ({size / (1024*1024):.1f} MB)")
            else:
                print(f"  {rel}  ({size:,} bytes)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
