#!/usr/bin/env python3
"""
prepare_tokenizer.py - Download Qwen2.5 tokenizers and generate VibeVoice special token mappings.

Downloads tokenizer.json from HuggingFace for 3 Qwen2.5 sizes (0.5b, 1.5b, 7b)
and creates special_tokens.json with VibeVoice-specific token ID mappings.

Usage:
    python scripts/prepare_tokenizer.py --output-dir tokenizer
"""

import argparse
import json
import os
import sys


def download_tokenizer(model_id: str, output_dir: str) -> str:
    """Download tokenizer.json from HuggingFace hub."""
    from transformers import AutoTokenizer

    print(f"  Downloading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    os.makedirs(output_dir, exist_ok=True)

    # Save the tokenizer - this produces tokenizer.json and other files
    tokenizer.save_pretrained(output_dir)

    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_json_path):
        raise FileNotFoundError(f"tokenizer.json not created at {tokenizer_json_path}")

    # Clean up unnecessary files, keep only tokenizer.json
    for fname in os.listdir(output_dir):
        if fname != "tokenizer.json" and fname != "special_tokens.json":
            fpath = os.path.join(output_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)

    return tokenizer_json_path


def extract_special_tokens_tts(tokenizer_json_path: str) -> dict:
    """Extract VibeVoice TTS special token IDs from tokenizer.json.

    TTS models reuse Qwen2-VL vision tokens:
      speech_start  = <|vision_start|>
      speech_end    = <|vision_end|>
      speech_diffusion = <|vision_pad|>
      eos           = <|endoftext|>
      pad           = <|image_pad|>
    """
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    # Build token -> id mapping from the vocab in added_tokens
    token_to_id = {}
    if "added_tokens" in tok_data:
        for entry in tok_data["added_tokens"]:
            token_to_id[entry["content"]] = entry["id"]

    # Also check the main model vocab
    if "model" in tok_data and "vocab" in tok_data["model"]:
        vocab = tok_data["model"]["vocab"]
        for token, tid in vocab.items():
            if token not in token_to_id:
                token_to_id[token] = tid

    # TTS special tokens
    tts_mapping = {
        "speech_start": "<|vision_start|>",
        "speech_end": "<|vision_end|>",
        "speech_diffusion": "<|vision_pad|>",
        "eos": "<|endoftext|>",
        "pad": "<|image_pad|>",
    }

    result = {"model_type": "tts"}
    for key, token_str in tts_mapping.items():
        tid = token_to_id.get(token_str)
        if tid is None:
            print(f"  WARNING: Token '{token_str}' not found in vocab for {key}")
            result[key] = {"token": token_str, "id": -1}
        else:
            result[key] = {"token": token_str, "id": tid}

    return result


def extract_special_tokens_asr(tokenizer_json_path: str) -> dict:
    """Extract VibeVoice ASR special token IDs from tokenizer.json.

    ASR model uses different tokens:
      speech_start  = <|object_ref_start|>
      speech_end    = <|object_ref_end|>
      speech_pad    = <|box_start|>
      eos           = <|endoftext|>
      pad           = <|image_pad|>
    """
    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)

    token_to_id = {}
    if "added_tokens" in tok_data:
        for entry in tok_data["added_tokens"]:
            token_to_id[entry["content"]] = entry["id"]

    if "model" in tok_data and "vocab" in tok_data["model"]:
        vocab = tok_data["model"]["vocab"]
        for token, tid in vocab.items():
            if token not in token_to_id:
                token_to_id[token] = tid

    asr_mapping = {
        "speech_start": "<|object_ref_start|>",
        "speech_end": "<|object_ref_end|>",
        "speech_pad": "<|box_start|>",
        "eos": "<|endoftext|>",
        "pad": "<|image_pad|>",
    }

    result = {"model_type": "asr"}
    for key, token_str in asr_mapping.items():
        tid = token_to_id.get(token_str)
        if tid is None:
            print(f"  WARNING: Token '{token_str}' not found in vocab for {key}")
            result[key] = {"token": token_str, "id": -1}
        else:
            result[key] = {"token": token_str, "id": tid}

    return result


def main():
    parser = argparse.ArgumentParser(description="Prepare Qwen2.5 tokenizers for VibeVoice")
    parser.add_argument(
        "--output-dir", type=str, default="tokenizer",
        help="Output directory for tokenizer files (default: tokenizer)",
    )
    args = parser.parse_args()

    # Model configurations: (size_name, HuggingFace model_id, special_token_type)
    # 0.5b is used by VibeVoice-Realtime-0.5B (TTS streaming)
    # 1.5b is used by VibeVoice-1.5B (TTS full)
    # 7b is used by VibeVoice-ASR
    configs = [
        ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B", "tts"),
        ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B", "tts"),
        ("qwen2.5-7b", "Qwen/Qwen2.5-7B", "asr"),
    ]

    for size_name, model_id, token_type in configs:
        print(f"\n{'='*60}")
        print(f"Processing {size_name} ({model_id})")
        print(f"{'='*60}")

        out_dir = os.path.join(args.output_dir, size_name)
        os.makedirs(out_dir, exist_ok=True)

        # Step 1: Download tokenizer
        tokenizer_json_path = download_tokenizer(model_id, out_dir)
        print(f"  Saved tokenizer.json to {tokenizer_json_path}")

        # Step 2: Extract special token mappings
        if token_type == "tts":
            special_tokens = extract_special_tokens_tts(tokenizer_json_path)
        else:
            special_tokens = extract_special_tokens_asr(tokenizer_json_path)

        # Step 3: Save special_tokens.json
        special_tokens_path = os.path.join(out_dir, "special_tokens.json")
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(special_tokens, f, indent=2, ensure_ascii=False)
        print(f"  Saved special_tokens.json to {special_tokens_path}")

        # Print summary
        print(f"  Special tokens ({token_type}):")
        for key, val in special_tokens.items():
            if key == "model_type":
                continue
            print(f"    {key}: {val['token']} -> id={val['id']}")

    print(f"\nDone! Tokenizer files saved to {args.output_dir}/")
    print(f"Directory structure:")
    for size_name, _, _ in configs:
        d = os.path.join(args.output_dir, size_name)
        if os.path.exists(d):
            files = os.listdir(d)
            for f in sorted(files):
                fpath = os.path.join(d, f)
                size = os.path.getsize(fpath)
                print(f"  {size_name}/{f}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
