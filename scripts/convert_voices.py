#!/usr/bin/env python3
"""
convert_voices.py - Convert voice presets for C++ server consumption.

Streaming model voices (.pt KV-cache) -> binary .bin format
Full model voices (.wav) -> copied as-is

Binary format for streaming voices:
  Header (32 bytes):
    u32 magic          = 0x56425643 ("VBVC")
    u32 version        = 1
    u32 num_groups     = 2 (lm, tts_lm)
    u32 hidden_size    = 896
    u32 head_dim       = 64
    u32 reserved[3]    = 0

  For each group ("lm", "tts_lm"):
    Group header (16 bytes):
      u32 num_layers
      u32 num_kv_heads
      u32 seq_len
      u32 reserved = 0
    Body:
      last_hidden_state: fp16 [seq_len, hidden_size]
      For each layer:
        key_cache:   fp16 [num_kv_heads, seq_len, head_dim]
        value_cache: fp16 [num_kv_heads, seq_len, head_dim]

Usage:
    python scripts/convert_voices.py --voices-dir <path>/voices --output-dir voices
"""

import argparse
import os
import shutil
import struct
import sys

import numpy as np
import torch


MAGIC = 0x56425643  # "VBVC"
VERSION = 1


def convert_streaming_voice(pt_path: str, bin_path: str) -> dict:
    """Convert a streaming model .pt voice preset to binary format.

    The .pt file contains a dict with keys "lm" and "tts_lm",
    each being a BaseModelOutputWithPast with:
      - last_hidden_state: [1, seq_len, hidden_size] (bfloat16)
      - past_key_values: DynamicCache with key_cache and value_cache lists
        Each entry: [1, num_kv_heads, seq_len, head_dim] (bfloat16)
    """
    print(f"  Loading {pt_path}...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    groups = []
    group_names = ["lm", "tts_lm"]

    for gname in group_names:
        if gname not in data:
            raise ValueError(f"Expected key '{gname}' in .pt file, got keys: {list(data.keys())}")

        group_data = data[gname]
        lhs = group_data.last_hidden_state  # [1, seq_len, hidden_size]
        kv_cache = group_data.past_key_values

        # Extract cache lists
        if hasattr(kv_cache, "key_cache"):
            key_cache_list = kv_cache.key_cache
            value_cache_list = kv_cache.value_cache
        else:
            # Fallback: might be a list of tuples
            key_cache_list = [kv[0] for kv in kv_cache]
            value_cache_list = [kv[1] for kv in kv_cache]

        num_layers = len(key_cache_list)
        # Shapes: key/value [1, num_kv_heads, seq_len, head_dim]
        num_kv_heads = key_cache_list[0].shape[1]
        seq_len = key_cache_list[0].shape[2]
        head_dim = key_cache_list[0].shape[3]
        hidden_size = lhs.shape[2]

        groups.append({
            "name": gname,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "last_hidden_state": lhs.squeeze(0).to(torch.float16).numpy(),  # [seq_len, hidden_size]
            "key_cache": [k.squeeze(0).to(torch.float16).numpy() for k in key_cache_list],  # [num_kv_heads, seq_len, head_dim]
            "value_cache": [v.squeeze(0).to(torch.float16).numpy() for v in value_cache_list],
        })

    # Validate consistent hidden_size and head_dim across groups
    hidden_size = groups[0]["hidden_size"]
    head_dim = groups[0]["head_dim"]
    for g in groups:
        assert g["hidden_size"] == hidden_size, f"Hidden size mismatch: {g['hidden_size']} vs {hidden_size}"
        assert g["head_dim"] == head_dim, f"Head dim mismatch: {g['head_dim']} vs {head_dim}"

    # Write binary
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)

    with open(bin_path, "wb") as f:
        # File header (32 bytes)
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(groups)))
        f.write(struct.pack("<I", hidden_size))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", 0))  # reserved
        f.write(struct.pack("<I", 0))  # reserved
        f.write(struct.pack("<I", 0))  # reserved

        total_data_bytes = 0

        for g in groups:
            # Group header (16 bytes)
            f.write(struct.pack("<I", g["num_layers"]))
            f.write(struct.pack("<I", g["num_kv_heads"]))
            f.write(struct.pack("<I", g["seq_len"]))
            f.write(struct.pack("<I", 0))  # reserved

            # last_hidden_state: fp16 [seq_len, hidden_size]
            lhs_bytes = g["last_hidden_state"].tobytes()
            f.write(lhs_bytes)
            total_data_bytes += len(lhs_bytes)

            # KV cache: for each layer, key then value
            for layer_idx in range(g["num_layers"]):
                k_bytes = g["key_cache"][layer_idx].tobytes()
                v_bytes = g["value_cache"][layer_idx].tobytes()
                f.write(k_bytes)
                f.write(v_bytes)
                total_data_bytes += len(k_bytes) + len(v_bytes)

    file_size = os.path.getsize(bin_path)
    info = {
        "file_size": file_size,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "groups": [(g["name"], g["num_layers"], g["num_kv_heads"], g["seq_len"]) for g in groups],
    }
    return info


def main():
    parser = argparse.ArgumentParser(description="Convert VibeVoice voice presets")
    parser.add_argument(
        "--voices-dir", type=str, required=True,
        help="Path to source voices directory (containing streaming_model/ and full_model/)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="voices",
        help="Output directory (default: voices)",
    )
    args = parser.parse_args()

    streaming_src = os.path.join(args.voices_dir, "streaming_model")
    full_src = os.path.join(args.voices_dir, "full_model")
    streaming_dst = os.path.join(args.output_dir, "streaming_model")
    full_dst = os.path.join(args.output_dir, "full_model")

    # Convert streaming model voices (.pt -> .bin)
    if os.path.isdir(streaming_src):
        print(f"\n{'='*60}")
        print(f"Converting streaming model voices")
        print(f"  Source: {streaming_src}")
        print(f"  Output: {streaming_dst}")
        print(f"{'='*60}")

        os.makedirs(streaming_dst, exist_ok=True)
        pt_files = sorted([f for f in os.listdir(streaming_src) if f.endswith(".pt")])

        if not pt_files:
            print("  No .pt files found!")
        else:
            print(f"  Found {len(pt_files)} voice presets")

        for pt_file in pt_files:
            bin_name = pt_file.replace(".pt", ".bin")
            pt_path = os.path.join(streaming_src, pt_file)
            bin_path = os.path.join(streaming_dst, bin_name)

            try:
                info = convert_streaming_voice(pt_path, bin_path)
                print(f"  {pt_file} -> {bin_name}")
                print(f"    Size: {info['file_size']:,} bytes")
                print(f"    Hidden: {info['hidden_size']}, HeadDim: {info['head_dim']}")
                for gname, nlayers, nheads, seqlen in info["groups"]:
                    print(f"    {gname}: {nlayers} layers, {nheads} kv_heads, seq_len={seqlen}")
            except Exception as e:
                print(f"  ERROR converting {pt_file}: {e}", file=sys.stderr)
                raise
    else:
        print(f"No streaming_model directory found at {streaming_src}, skipping.")

    # Copy full model voices (.wav -> .wav)
    if os.path.isdir(full_src):
        print(f"\n{'='*60}")
        print(f"Copying full model voices")
        print(f"  Source: {full_src}")
        print(f"  Output: {full_dst}")
        print(f"{'='*60}")

        os.makedirs(full_dst, exist_ok=True)
        wav_files = sorted([f for f in os.listdir(full_src) if f.endswith(".wav")])

        if not wav_files:
            print("  No .wav files found!")
        else:
            print(f"  Found {len(wav_files)} voice presets")

        for wav_file in wav_files:
            src_path = os.path.join(full_src, wav_file)
            dst_path = os.path.join(full_dst, wav_file)
            shutil.copy2(src_path, dst_path)
            size = os.path.getsize(dst_path)
            print(f"  {wav_file} ({size:,} bytes)")
    else:
        print(f"No full_model directory found at {full_src}, skipping.")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done! Voice presets saved to {args.output_dir}/")
    print(f"{'='*60}")

    for subdir in ["streaming_model", "full_model"]:
        d = os.path.join(args.output_dir, subdir)
        if os.path.isdir(d):
            files = os.listdir(d)
            total_size = sum(os.path.getsize(os.path.join(d, f)) for f in files)
            print(f"  {subdir}/: {len(files)} files, {total_size:,} bytes total")


if __name__ == "__main__":
    main()
