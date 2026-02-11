#!/usr/bin/env python3
"""
export_streaming_onnx.py - Export streaming versions of semantic encoder and
acoustic decoder with packed cache I/O for TensorRT inference.

Each model gets two additional tensors:
  - cache_in  [1, total_cache_elements]  (flat-packed layer caches)
  - cache_out [1, total_cache_elements]  (updated caches after one frame)

Usage:
    python scripts/export_streaming_onnx.py \
        --model-dir C:/Users/onion/Desktop/Workspace/models/VibeVoice-1.5B \
        --output-dir onnx/tts_1.5b \
        --vibevoice-lib C:/Users/onion/Desktop/Workspace/VibeVoice
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Cache specification helpers
# ---------------------------------------------------------------------------

def build_encoder_cache_spec(encoder):
    """Build ordered list of (offset, channels, context_size, layer_type)
    for every cacheable layer in a TokenizerEncoder, following the exact
    iteration order of forward_features + head."""
    spec = []
    offset = 0

    for i in range(len(encoder.depths)):
        # --- downsample / stem layer (nn.Sequential wrapping SConv1d) ---
        sconv = encoder.downsample_layers[i][0]
        ch = sconv.in_channels
        ctx = sconv.context_size
        spec.append((offset, ch, ctx, "sconv1d"))
        offset += ch * ctx

        # --- stage blocks ---
        for block in encoder.stages[i]:
            blk_sconv = block.mixer.conv       # SConv1d (depthwise)
            ch_b = blk_sconv.in_channels
            ctx_b = blk_sconv.context_size
            spec.append((offset, ch_b, ctx_b, "sconv1d"))
            offset += ch_b * ctx_b

    # --- head (SConv1d) ---
    head_sconv = encoder.head
    ch_h = head_sconv.in_channels
    ctx_h = head_sconv.context_size
    spec.append((offset, ch_h, ctx_h, "sconv1d"))
    offset += ch_h * ctx_h

    return spec, offset


def build_decoder_cache_spec(decoder):
    """Build ordered list of (offset, channels, context_size, layer_type)
    for every cacheable layer in a TokenizerDecoder."""
    spec = []
    offset = 0

    for i in range(len(decoder.depths)):
        # --- upsample / stem layer ---
        layer = decoder.upsample_layers[i][0]
        ch = layer.in_channels
        ctx = layer.context_size
        from vibevoice.modular.modular_vibevoice_tokenizer import SConvTranspose1d
        ltype = "sconvtr1d" if isinstance(layer, SConvTranspose1d) else "sconv1d"
        spec.append((offset, ch, ctx, ltype))
        offset += ch * ctx

        # --- stage blocks ---
        for block in decoder.stages[i]:
            blk_sconv = block.mixer.conv
            ch_b = blk_sconv.in_channels
            ctx_b = blk_sconv.context_size
            spec.append((offset, ch_b, ctx_b, "sconv1d"))
            offset += ch_b * ctx_b

    # --- head (SConv1d) ---
    head_sconv = decoder.head
    ch_h = head_sconv.in_channels
    ctx_h = head_sconv.context_size
    spec.append((offset, ch_h, ctx_h, "sconv1d"))
    offset += ch_h * ctx_h

    return spec, offset


# ---------------------------------------------------------------------------
# Streaming Wrappers
# ---------------------------------------------------------------------------

class StreamingSemanticEncoderWrapper(nn.Module):
    """Wraps a TokenizerEncoder for streaming ONNX export.

    Input:  audio [1, 1, 3200]  +  cache_in [1, total_cache_size]
    Output: semantic_mean [1, 1, vae_dim]  +  cache_out [1, total_cache_size]
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.cache_spec, self.total_cache_size = build_encoder_cache_spec(encoder)
        print(f"  Semantic encoder: {len(self.cache_spec)} cache entries, "
              f"{self.total_cache_size} elements ({self.total_cache_size * 4 / 1024:.1f} KB fp32)")

    def forward(self, audio, cache_in):
        x = audio
        new_caches = []
        cache_idx = 0

        for i in range(len(self.encoder.depths)):
            # --- downsample / stem ---
            sconv = self.encoder.downsample_layers[i][0]
            offset, ch, ctx, _ = self.cache_spec[cache_idx]
            layer_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)
            x_with_ctx = torch.cat([layer_cache, x], dim=2)
            x = sconv.conv(x_with_ctx)                       # NormConv1d
            new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1) if ctx > 0
                              else x_with_ctx[:, :, :0].reshape(1, -1))
            cache_idx += 1

            # --- stage blocks ---
            for block in self.encoder.stages[i]:
                offset, ch, ctx, _ = self.cache_spec[cache_idx]
                residual = x
                x = block.norm(x)
                blk_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)
                x_with_ctx = torch.cat([blk_cache, x], dim=2)
                x = block.mixer.conv.conv(x_with_ctx)         # NormConv1d
                new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1))
                cache_idx += 1
                if block.gamma is not None:
                    x = x * block.gamma.unsqueeze(-1)
                x = residual + x
                # FFN
                residual = x
                x = block.ffn_norm(x)
                x = x.permute(0, 2, 1)
                x = block.ffn(x)
                x = x.permute(0, 2, 1)
                if block.ffn_gamma is not None:
                    x = x * block.ffn_gamma.unsqueeze(-1)
                x = residual + x

        # norm (Identity when disable_last_norm=True)
        x = self.encoder.norm(x)

        # --- head ---
        offset, ch, ctx, _ = self.cache_spec[cache_idx]
        head_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)
        x_with_ctx = torch.cat([head_cache, x], dim=2)
        x = self.encoder.head.conv(x_with_ctx)               # NormConv1d
        new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1))

        cache_out = torch.cat(new_caches, dim=1)

        # [1, vae_dim, 1] -> [1, 1, vae_dim]
        x = x.permute(0, 2, 1)
        return x, cache_out


class StreamingAcousticDecoderWrapper(nn.Module):
    """Wraps a TokenizerDecoder for streaming ONNX export.

    Input:  latent [1, 64, 1]  +  cache_in [1, total_cache_size]
    Output: audio  [1, 1, 3200] +  cache_out [1, total_cache_size]
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.cache_spec, self.total_cache_size = build_decoder_cache_spec(decoder)
        print(f"  Acoustic decoder: {len(self.cache_spec)} cache entries, "
              f"{self.total_cache_size} elements ({self.total_cache_size * 4 / 1024:.1f} KB fp32)")

    def _sconvtr_streaming(self, sconvtr, x, tr_cache, T_new):
        """Run a single SConvTranspose1d layer in streaming mode.

        Returns (output, new_cache_flat).
        """
        full_input = torch.cat([tr_cache, x], dim=2)
        full_output = sconvtr.convtr(full_input)              # NormConvTranspose1d

        # Remove causal right padding (trim_right_ratio = 1.0)
        padding_total = sconvtr.kernel_size - sconvtr.stride
        if padding_total > 0:
            full_output = full_output[:, :, :-padding_total]

        # Extract only new output: last T_new * stride samples
        expected_new = T_new * sconvtr.stride
        output = full_output[:, :, -expected_new:]

        # Updated cache: last context_size elements of full_input
        ctx = sconvtr.context_size
        new_cache = full_input[:, :, -ctx:]
        return output, new_cache.reshape(1, -1)

    def forward(self, latent, cache_in):
        x = latent
        new_caches = []
        cache_idx = 0

        for i in range(len(self.decoder.depths)):
            # --- upsample / stem ---
            layer = self.decoder.upsample_layers[i][0]
            offset, ch, ctx, ltype = self.cache_spec[cache_idx]
            layer_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)

            if ltype == "sconvtr1d":
                T_new = x.shape[2]
                x, nc = self._sconvtr_streaming(layer, x, layer_cache, T_new)
                new_caches.append(nc)
            else:
                # SConv1d (stem)
                x_with_ctx = torch.cat([layer_cache, x], dim=2)
                x = layer.conv(x_with_ctx)
                new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1) if ctx > 0
                                  else x_with_ctx[:, :, :0].reshape(1, -1))
            cache_idx += 1

            # --- stage blocks ---
            for block in self.decoder.stages[i]:
                offset, ch, ctx, _ = self.cache_spec[cache_idx]
                residual = x
                x = block.norm(x)
                blk_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)
                x_with_ctx = torch.cat([blk_cache, x], dim=2)
                x = block.mixer.conv.conv(x_with_ctx)
                new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1))
                cache_idx += 1
                if block.gamma is not None:
                    x = x * block.gamma.unsqueeze(-1)
                x = residual + x
                # FFN
                residual = x
                x = block.ffn_norm(x)
                x = x.permute(0, 2, 1)
                x = block.ffn(x)
                x = x.permute(0, 2, 1)
                if block.ffn_gamma is not None:
                    x = x * block.ffn_gamma.unsqueeze(-1)
                x = residual + x

        # norm (Identity)
        x = self.decoder.norm(x)

        # --- head ---
        offset, ch, ctx, _ = self.cache_spec[cache_idx]
        head_cache = cache_in[:, offset:offset + ch * ctx].reshape(1, ch, ctx)
        x_with_ctx = torch.cat([head_cache, x], dim=2)
        x = self.decoder.head.conv(x_with_ctx)
        new_caches.append(x_with_ctx[:, :, -ctx:].reshape(1, -1))

        cache_out = torch.cat(new_caches, dim=1)
        return x, cache_out


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_streaming_vs_full(model, device="cpu"):
    """Verify that streaming encoder output matches full (non-streaming) output."""
    from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache

    print("\n=== Verifying streaming vs full output ===")

    # Generate 10 frames of random audio
    n_frames = 10
    hop = 3200
    audio = torch.randn(1, 1, n_frames * hop, device=device)

    # 1) Full (non-streaming) encode
    sem_full = model.model.semantic_tokenizer.encode(audio).mean  # [1, 10, 128]

    # 2) Streaming encode
    sem_cache = VibeVoiceTokenizerStreamingCache()
    sample_idx = torch.tensor([0], device=device)
    sem_chunks = []
    for t in range(n_frames):
        chunk = audio[:, :, t * hop:(t + 1) * hop]
        feat = model.model.semantic_tokenizer.encode(
            chunk, cache=sem_cache, sample_indices=sample_idx, use_cache=True
        ).mean
        sem_chunks.append(feat)
    sem_stream = torch.cat(sem_chunks, dim=1)

    diff = (sem_full - sem_stream).abs()
    print(f"  Semantic encoder: max_diff={diff.max().item():.6e}, "
          f"mean_diff={diff.mean().item():.6e}")
    if diff.max().item() > 1e-3:
        print("  WARNING: Large difference detected!")
    else:
        print("  OK: Streaming output matches full output.")

    # 3) Full (non-streaming) decode
    dummy_latent = torch.randn(1, 64, n_frames, device=device)
    audio_full = model.model.acoustic_tokenizer.decode(dummy_latent)

    # 4) Streaming decode
    ac_cache = VibeVoiceTokenizerStreamingCache()
    audio_chunks = []
    for t in range(n_frames):
        lat_chunk = dummy_latent[:, :, t:t + 1]
        aud = model.model.acoustic_tokenizer.decode(
            lat_chunk, cache=ac_cache, sample_indices=sample_idx, use_cache=True
        )
        audio_chunks.append(aud)
    audio_stream = torch.cat(audio_chunks, dim=-1)

    diff_a = (audio_full - audio_stream).abs()
    print(f"  Acoustic decoder: max_diff={diff_a.max().item():.6e}, "
          f"mean_diff={diff_a.mean().item():.6e}")
    if diff_a.max().item() > 1e-3:
        print("  WARNING: Large difference detected!")
    else:
        print("  OK: Streaming output matches full output.")

    return diff.max().item(), diff_a.max().item()


def verify_onnx_vs_pytorch(onnx_path, wrapper, dummy_inputs, label="model"):
    """Compare ONNX runtime output with PyTorch wrapper output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  Skipping ONNX verification (onnxruntime not installed)")
        return

    print(f"\n=== Verifying ONNX vs PyTorch: {label} ===")

    # PyTorch
    with torch.no_grad():
        pt_out = wrapper(*dummy_inputs)
    if isinstance(pt_out, tuple):
        pt_features = pt_out[0].numpy()
        pt_cache = pt_out[1].numpy()
    else:
        pt_features = pt_out.numpy()
        pt_cache = None

    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path)
    ort_inputs = {}
    for inp, tensor in zip(sess.get_inputs(), dummy_inputs):
        ort_inputs[inp.name] = tensor.numpy()
    ort_out = sess.run(None, ort_inputs)

    diff_feat = np.abs(pt_features - ort_out[0]).max()
    print(f"  Features max diff: {diff_feat:.6e}")
    if pt_cache is not None and len(ort_out) > 1:
        diff_cache = np.abs(pt_cache - ort_out[1]).max()
        print(f"  Cache max diff:    {diff_cache:.6e}")

    if diff_feat < 1e-4:
        print(f"  OK: {label} ONNX matches PyTorch.")
    else:
        print(f"  WARNING: Large diff in {label}!")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx_model(wrapper, dummy_inputs, output_path, input_names, output_names,
                      dynamic_axes=None, opset_version=17):
    """Export a wrapped module to ONNX with validation."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"  Exporting to {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx.checker.check_model(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  OK: {output_path} ({size_mb:.1f} MB)")


def load_model(model_dir, model_type="tts_1.5b"):
    """Load VibeVoice model (1.5B or 0.5B)."""
    vibevoice_lib = os.environ.get("VIBEVOICE_LIB")
    if vibevoice_lib:
        sys.path.insert(0, vibevoice_lib)

    import vibevoice  # noqa: F401

    print(f"Loading model from {model_dir} (type={model_type})...")
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    if model_type == "tts_0.5b":
        from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        model_config = VibeVoiceStreamingConfig.from_pretrained(model_dir)
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch.float32
        )
    else:
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        model_config = VibeVoiceConfig.from_pretrained(model_dir)
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch.float32
        )

    model.eval()
    print(f"Model loaded: {type(model).__name__}")
    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Export streaming semantic encoder & acoustic decoder to ONNX"
    )
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to VibeVoice model directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for ONNX files (e.g. onnx/tts_1.5b)")
    parser.add_argument("--vibevoice-lib", type=str, default=None,
                        help="Path to VibeVoice Python library")
    parser.add_argument("--model-type", type=str, default="tts_1.5b",
                        choices=["tts_1.5b", "tts_0.5b"],
                        help="Model type (default: tts_1.5b)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip streaming vs full verification")
    parser.add_argument("--only", type=str, nargs="+", default=None,
                        choices=["semantic_encoder", "acoustic_decoder"],
                        help="Only export specified components")
    args = parser.parse_args()

    if args.vibevoice_lib:
        os.environ["VIBEVOICE_LIB"] = args.vibevoice_lib

    os.makedirs(args.output_dir, exist_ok=True)

    model, config = load_model(args.model_dir, args.model_type)

    # --- Step 0: Verify streaming vs full ---
    # Skip semantic encoder verification for 0.5B (no semantic_tokenizer)
    if not args.skip_verify and args.model_type != "tts_0.5b":
        verify_streaming_vs_full(model)

    hop = 3200  # hop_length

    should_all = args.only is None

    # 0.5B has no semantic_tokenizer; auto-skip semantic_encoder
    has_semantic = hasattr(model.model, "semantic_tokenizer")

    with torch.no_grad():
        # --- Step 1: Streaming Semantic Encoder ---
        if (should_all or "semantic_encoder" in args.only) and has_semantic:
            print("\n" + "=" * 60)
            print("Exporting streaming_semantic_encoder")
            print("=" * 60)

            sem_encoder = model.model.semantic_tokenizer.encoder
            sem_wrapper = StreamingSemanticEncoderWrapper(sem_encoder)
            sem_wrapper.eval()

            dummy_audio = torch.randn(1, 1, hop)
            dummy_cache = torch.zeros(1, sem_wrapper.total_cache_size)

            # Test forward pass
            print("  Testing forward pass...")
            out, cout = sem_wrapper(dummy_audio, dummy_cache)
            print(f"  Output shape: {out.shape}, cache shape: {cout.shape}")
            assert out.shape == (1, 1, 128) or out.shape[1] == 1, \
                f"Unexpected output shape: {out.shape}"
            assert cout.shape == dummy_cache.shape, \
                f"Cache shape mismatch: {cout.shape} vs {dummy_cache.shape}"

            sem_onnx_path = os.path.join(args.output_dir, "streaming_semantic_encoder.onnx")
            export_onnx_model(
                sem_wrapper,
                (dummy_audio, dummy_cache),
                sem_onnx_path,
                input_names=["audio", "cache_in"],
                output_names=["semantic_mean", "cache_out"],
            )

            # Verify ONNX vs PyTorch
            verify_onnx_vs_pytorch(
                sem_onnx_path, sem_wrapper,
                (dummy_audio, dummy_cache),
                label="streaming_semantic_encoder",
            )

            # Save cache metadata
            meta = {
                "total_cache_size": sem_wrapper.total_cache_size,
                "num_layers": len(sem_wrapper.cache_spec),
                "layers": [
                    {"offset": o, "channels": c, "context_size": cs, "type": t}
                    for o, c, cs, t in sem_wrapper.cache_spec
                ],
            }
            meta_path = os.path.join(args.output_dir, "streaming_semantic_encoder_cache.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  Cache metadata saved to {meta_path}")

        # --- Step 2: Streaming Acoustic Decoder ---
        if should_all or "acoustic_decoder" in args.only:
            print("\n" + "=" * 60)
            print("Exporting streaming_acoustic_decoder")
            print("=" * 60)

            ac_decoder = model.model.acoustic_tokenizer.decoder
            ac_wrapper = StreamingAcousticDecoderWrapper(ac_decoder)
            ac_wrapper.eval()

            vae_dim = config.get("acoustic_vae_dim", 64)
            dummy_latent = torch.randn(1, vae_dim, 1)
            dummy_cache = torch.zeros(1, ac_wrapper.total_cache_size)

            # Test forward pass
            print("  Testing forward pass...")
            out, cout = ac_wrapper(dummy_latent, dummy_cache)
            print(f"  Output shape: {out.shape}, cache shape: {cout.shape}")
            assert out.shape == (1, 1, hop), \
                f"Unexpected output shape: {out.shape}, expected [1, 1, {hop}]"
            assert cout.shape == dummy_cache.shape, \
                f"Cache shape mismatch: {cout.shape} vs {dummy_cache.shape}"

            ac_onnx_path = os.path.join(args.output_dir, "streaming_acoustic_decoder.onnx")
            export_onnx_model(
                ac_wrapper,
                (dummy_latent, dummy_cache),
                ac_onnx_path,
                input_names=["latent", "cache_in"],
                output_names=["audio", "cache_out"],
            )

            # Verify ONNX vs PyTorch
            verify_onnx_vs_pytorch(
                ac_onnx_path, ac_wrapper,
                (dummy_latent, dummy_cache),
                label="streaming_acoustic_decoder",
            )

            # Save cache metadata
            meta = {
                "total_cache_size": ac_wrapper.total_cache_size,
                "num_layers": len(ac_wrapper.cache_spec),
                "layers": [
                    {"offset": o, "channels": c, "context_size": cs, "type": t}
                    for o, c, cs, t in ac_wrapper.cache_spec
                ],
            }
            meta_path = os.path.join(args.output_dir, "streaming_acoustic_decoder_cache.json")
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  Cache metadata saved to {meta_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("Streaming ONNX export complete!")
    print(f"Output: {args.output_dir}")
    print(f"{'=' * 60}")
    for fname in sorted(os.listdir(args.output_dir)):
        if "streaming" in fname:
            fpath = os.path.join(args.output_dir, fname)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"  {fname}  ({size / (1024*1024):.1f} MB)")
            else:
                print(f"  {fname}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
