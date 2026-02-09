#!/usr/bin/env python3
"""
export_onnx.py - Export VibeVoice sub-models to ONNX format.

Exports 5 sub-models (acoustic_encoder, acoustic_decoder, semantic_encoder,
language_model, diffusion_head) from safetensors weights to individual ONNX files.
Connector weights (small MLPs) are saved as raw binary for C++ direct implementation.

Usage:
    python scripts/export_onnx.py --model-dir <path> --model-type <tts_1.5b|tts_0.5b|asr> --output-dir onnx/<variant>

Examples:
    python scripts/export_onnx.py --model-dir C:/models/VibeVoice-1.5B --model-type tts_1.5b --output-dir onnx/tts_1.5b
    python scripts/export_onnx.py --model-dir C:/models/VibeVoice-Realtime-0.5B --model-type tts_0.5b --output-dir onnx/tts_0.5b
    python scripts/export_onnx.py --model-dir C:/models/VibeVoice-ASR --model-type asr --output-dir onnx/asr
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn


# ============================================================================
# Wrapper modules for clean ONNX export (no cache, no sampling)
# ============================================================================

class AcousticEncoderWrapper(nn.Module):
    """Wraps acoustic tokenizer encoder for ONNX export.
    Input: audio [B, 1, N]
    Output: latent_mean [B, T, vae_dim] where T = N / hop_length
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio):
        # encoder expects [B, channels, T], returns [B, vae_dim, T'] internally
        # The actual encoder output needs permutation to [B, T', vae_dim]
        out = self.encoder(audio)
        # encoder output shape: [B, vae_dim, T] -> permute to [B, T, vae_dim]
        if out.dim() == 3 and out.shape[1] != out.shape[2]:
            # Check if vae_dim is in dim=1 (typical for conv output)
            # vae_dim is usually 64 or 128, much smaller than T
            if out.shape[1] < out.shape[2]:
                out = out.permute(0, 2, 1)
        return out


class AcousticDecoderWrapper(nn.Module):
    """Wraps acoustic tokenizer decoder for ONNX export.
    Input: latent [B, 64, T] (vae_dim-first format, matching decoder's internal convention)
    Output: audio [B, 1, M] where M = T * hop_length
    """
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, latent):
        # decoder expects [B, vae_dim, T] format internally
        audio = self.decoder(latent)
        return audio


class SemanticEncoderWrapper(nn.Module):
    """Wraps semantic tokenizer encoder for ONNX export.
    Input: audio [B, 1, N]
    Output: semantic_mean [B, T, 128]
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, audio):
        out = self.encoder(audio)
        if out.dim() == 3 and out.shape[1] < out.shape[2]:
            out = out.permute(0, 2, 1)
        return out


class DiffusionHeadWrapper(nn.Module):
    """Wraps diffusion head for ONNX export.
    Input: noisy_images [B, 64], timesteps [B], condition [B, H]
    Output: v_prediction [B, 64]
    """
    def __init__(self, prediction_head):
        super().__init__()
        self.prediction_head = prediction_head

    def forward(self, noisy_images, timesteps, condition):
        # TimestepEmbedder returns embedding.to(t.dtype), so if timesteps
        # is int64 the output would be int64 which breaks Linear layers.
        # Cast to float first.
        timesteps = timesteps.float()
        return self.prediction_head(noisy_images, timesteps, condition)


def _extract_kv_cache(past_key_values):
    """Extract KV cache tensors from various cache formats."""
    kv_outputs = []
    if past_key_values is not None:
        if hasattr(past_key_values, "key_cache"):
            # DynamicCache
            for i in range(len(past_key_values.key_cache)):
                kv_outputs.append(past_key_values.key_cache[i])
                kv_outputs.append(past_key_values.value_cache[i])
        else:
            # Tuple of tuples
            for kv in past_key_values:
                kv_outputs.append(kv[0])
                kv_outputs.append(kv[1])
    return kv_outputs


def _build_kv_cache(past_kv_flat, num_layers):
    """Build DynamicCache from flat list of KV tensors."""
    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache()
    for i in range(num_layers):
        k = past_kv_flat[2 * i]
        v = past_kv_flat[2 * i + 1]
        past_key_values.update(k, v, i)
    return past_key_values


class LMPrefillWrapper(nn.Module):
    """Wraps Qwen2ForCausalLM-style model for prefill (no KV-cache input).
    Input: inputs_embeds [B, S, H], position_ids [B, S]
    Output: logits [B, S, V], kv_cache (list of key/value tensors)
    Used for models that have lm_head (1.5B TTS, ASR).
    """
    def __init__(self, language_model, lm_head):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds, position_ids):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        kv_outputs = _extract_kv_cache(outputs.past_key_values)
        return (logits,) + tuple(kv_outputs)


class LMDecodeWrapper(nn.Module):
    """Wraps Qwen2ForCausalLM-style model for decode (with KV-cache).
    Input: inputs_embeds [B, 1, H], position_ids [B, 1], past KV tensors
    Output: logits [B, 1, V], updated KV tensors
    Used for models that have lm_head (1.5B TTS, ASR).
    """
    def __init__(self, language_model, lm_head, num_layers, num_kv_heads, head_dim):
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, inputs_embeds, position_ids, *past_kv_flat):
        past_key_values = _build_kv_cache(past_kv_flat, self.num_layers)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        kv_outputs = _extract_kv_cache(outputs.past_key_values)
        return (logits,) + tuple(kv_outputs)


class LMHiddenPrefillWrapper(nn.Module):
    """Wraps Qwen2Model (no lm_head) for prefill.
    Input: inputs_embeds [B, S, H], position_ids [B, S]
    Output: hidden_states [B, S, H], kv_cache tensors
    Used for 0.5B split LM (base_lm and tts_lm output hidden states, not logits).
    """
    def __init__(self, language_model):
        super().__init__()
        self.language_model = language_model

    def forward(self, inputs_embeds, position_ids):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=True,
        )
        kv_outputs = _extract_kv_cache(outputs.past_key_values)
        return (outputs.last_hidden_state,) + tuple(kv_outputs)


class LMHiddenDecodeWrapper(nn.Module):
    """Wraps Qwen2Model (no lm_head) for decode.
    Input: inputs_embeds [B, 1, H], position_ids [B, 1], past KV tensors
    Output: hidden_states [B, 1, H], updated KV tensors
    Used for 0.5B split LM.
    """
    def __init__(self, language_model, num_layers, num_kv_heads, head_dim):
        super().__init__()
        self.language_model = language_model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, inputs_embeds, position_ids, *past_kv_flat):
        past_key_values = _build_kv_cache(past_kv_flat, self.num_layers)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        kv_outputs = _extract_kv_cache(outputs.past_key_values)
        return (outputs.last_hidden_state,) + tuple(kv_outputs)


# ============================================================================
# Weight extraction helpers
# ============================================================================

def save_linear_weights(module: nn.Module, path: str, name: str):
    """Save Linear layer weights as raw float16 binary.
    Format: [out_features, in_features] for weight, [out_features] for bias
    Header: u32 in_features, u32 out_features, u8 has_bias
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        if isinstance(module, nn.Linear):
            w = module.weight.data.cpu().to(torch.float16).numpy()
            has_bias = module.bias is not None
            f.write(struct.pack("<I", module.in_features))
            f.write(struct.pack("<I", module.out_features))
            f.write(struct.pack("<B", 1 if has_bias else 0))
            f.write(w.tobytes())
            if has_bias:
                b = module.bias.data.cpu().to(torch.float16).numpy()
                f.write(b.tobytes())
        else:
            raise ValueError(f"Expected nn.Linear, got {type(module)}")
    print(f"    Saved {name} -> {path} ({os.path.getsize(path):,} bytes)")


def save_connector_weights(connector, path: str, name: str):
    """Save SpeechConnector weights (fc1 + norm + fc2) as binary.
    Format:
      Header: u32 input_dim, u32 output_dim
      fc1.weight: fp16 [output_dim, input_dim]
      fc1.bias: fp16 [output_dim]
      norm.weight: fp16 [output_dim] (RMSNorm scale)
      fc2.weight: fp16 [output_dim, output_dim]
      fc2.bias: fp16 [output_dim]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        fc1 = connector.fc1
        norm = connector.norm
        fc2 = connector.fc2

        in_dim = fc1.in_features
        out_dim = fc1.out_features

        f.write(struct.pack("<I", in_dim))
        f.write(struct.pack("<I", out_dim))

        # fc1
        f.write(fc1.weight.data.cpu().to(torch.float16).numpy().tobytes())
        f.write(fc1.bias.data.cpu().to(torch.float16).numpy().tobytes())

        # RMSNorm weight
        f.write(norm.weight.data.cpu().to(torch.float16).numpy().tobytes())

        # fc2
        f.write(fc2.weight.data.cpu().to(torch.float16).numpy().tobytes())
        f.write(fc2.bias.data.cpu().to(torch.float16).numpy().tobytes())

    print(f"    Saved {name} -> {path} ({os.path.getsize(path):,} bytes)")


def save_scalar_buffer(tensor, path: str, name: str):
    """Save a scalar buffer as fp32 binary."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    val = tensor.cpu().float().numpy()
    with open(path, "wb") as f:
        f.write(val.tobytes())
    print(f"    Saved {name} -> {path} (value={val.item():.6f})")


def save_embedding_weights(embedding: nn.Embedding, path: str, name: str):
    """Save Embedding weights as binary.
    Header: u32 num_embeddings, u32 embedding_dim
    Body: fp16 [num_embeddings, embedding_dim]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", embedding.num_embeddings))
        f.write(struct.pack("<I", embedding.embedding_dim))
        f.write(embedding.weight.data.cpu().to(torch.float16).numpy().tobytes())
    print(f"    Saved {name} -> {path} ({os.path.getsize(path):,} bytes)")


def save_binary_classifier_weights(classifier, path: str, name: str):
    """Save BinaryClassifier weights (fc1 + fc2) as binary.
    Header: u32 hidden_size
    fc1.weight: fp16 [hidden_size, hidden_size]
    fc1.bias: fp16 [hidden_size]
    fc2.weight: fp16 [1, hidden_size]
    fc2.bias: fp16 [1]
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        hidden_size = classifier.fc1.in_features
        f.write(struct.pack("<I", hidden_size))
        # fc1
        f.write(classifier.fc1.weight.data.cpu().to(torch.float16).numpy().tobytes())
        f.write(classifier.fc1.bias.data.cpu().to(torch.float16).numpy().tobytes())
        # fc2
        f.write(classifier.fc2.weight.data.cpu().to(torch.float16).numpy().tobytes())
        f.write(classifier.fc2.bias.data.cpu().to(torch.float16).numpy().tobytes())
    print(f"    Saved {name} -> {path} ({os.path.getsize(path):,} bytes)")


# ============================================================================
# ONNX export functions
# ============================================================================

def export_onnx_model(wrapper, dummy_inputs, output_path, input_names, output_names,
                      dynamic_axes, opset_version=17):
    """Export a wrapped module to ONNX with validation."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"  Exporting to {output_path}...")
    # Use dynamo=False to force legacy TorchScript-based export,
    # which handles *args and dynamic_axes reliably.
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # Validate using path-based checking (avoids >2GiB protobuf limit
    # when external data format is used for large models)
    onnx.checker.check_model(output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  OK: {output_path} ({size_mb:.1f} MB)")


def get_lm_kv_names(num_layers, prefix=""):
    """Generate input/output names for KV cache tensors."""
    names = []
    for i in range(num_layers):
        names.append(f"{prefix}past_key_{i}")
        names.append(f"{prefix}past_value_{i}")
    return names


def get_present_kv_names(num_layers, prefix=""):
    """Generate output names for present KV cache tensors."""
    names = []
    for i in range(num_layers):
        names.append(f"{prefix}present_key_{i}")
        names.append(f"{prefix}present_value_{i}")
    return names


# ============================================================================
# Model loading
# ============================================================================

def load_model(model_dir: str, model_type: str):
    """Load a VibeVoice model from pretrained weights.

    Returns the loaded model and its config.
    """
    # Add the VibeVoice library to path so model classes are registered
    vibevoice_lib = os.environ.get("VIBEVOICE_LIB")
    if vibevoice_lib:
        sys.path.insert(0, vibevoice_lib)

    # Import VibeVoice to register model classes with AutoModel/AutoConfig
    import vibevoice  # noqa: F401

    print(f"Loading model from {model_dir} (type={model_type})...")

    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Use specific config/model classes to avoid AutoConfig lookup failures
    # (VibeVoice registers with AutoModel but not with AutoConfig's MODEL_MAPPING)
    if model_type == "tts_1.5b":
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
        model_config = VibeVoiceConfig.from_pretrained(model_dir)
        model = VibeVoiceForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch.float32
        )
    elif model_type == "tts_0.5b":
        from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        model_config = VibeVoiceStreamingConfig.from_pretrained(model_dir)
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch.float32
        )
    elif model_type == "asr":
        from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
        from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
        model_config = VibeVoiceASRConfig.from_pretrained(model_dir)
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            model_dir, config=model_config, torch_dtype=torch.float32
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    print(f"Model loaded successfully. Type: {type(model).__name__}")
    return model, config


# ============================================================================
# Export routines per model type
# ============================================================================

def export_acoustic_encoder(model, output_dir, model_type):
    """Export acoustic encoder to ONNX."""
    print(f"\n--- Exporting acoustic_encoder ---")

    if model_type == "tts_0.5b":
        print("  Skipping: 0.5B streaming model has no acoustic encoder weights")
        return

    encoder = model.model.acoustic_tokenizer.encoder

    wrapper = AcousticEncoderWrapper(encoder)
    wrapper.eval()

    # Dummy input: 1 second of audio at 24kHz
    dummy_audio = torch.randn(1, 1, 24000)

    export_onnx_model(
        wrapper, (dummy_audio,),
        os.path.join(output_dir, "acoustic_encoder.onnx"),
        input_names=["audio"],
        output_names=["latent_mean"],
        dynamic_axes={
            "audio": {0: "batch", 2: "audio_len"},
            "latent_mean": {0: "batch", 1: "seq_len"},
        },
    )


def export_acoustic_decoder(model, output_dir, model_type):
    """Export acoustic decoder to ONNX."""
    print(f"\n--- Exporting acoustic_decoder ---")

    if model_type == "asr":
        print("  Skipping: ASR model has no acoustic decoder")
        return

    if model_type == "tts_0.5b":
        decoder = model.model.acoustic_tokenizer.decoder
    else:
        decoder = model.model.acoustic_tokenizer.decoder

    wrapper = AcousticDecoderWrapper(decoder)
    wrapper.eval()

    # Dummy input: latent [B, 64, T] (vae_dim-first)
    dummy_latent = torch.randn(1, 64, 10)

    export_onnx_model(
        wrapper, (dummy_latent,),
        os.path.join(output_dir, "acoustic_decoder.onnx"),
        input_names=["latent"],
        output_names=["audio"],
        dynamic_axes={
            "latent": {0: "batch", 2: "latent_len"},
            "audio": {0: "batch", 2: "audio_len"},
        },
    )


def export_semantic_encoder(model, output_dir, model_type):
    """Export semantic encoder to ONNX."""
    print(f"\n--- Exporting semantic_encoder ---")

    if model_type == "tts_0.5b":
        print("  Skipping: 0.5B streaming model has no semantic encoder")
        return

    encoder = model.model.semantic_tokenizer.encoder
    wrapper = SemanticEncoderWrapper(encoder)
    wrapper.eval()

    dummy_audio = torch.randn(1, 1, 24000)

    export_onnx_model(
        wrapper, (dummy_audio,),
        os.path.join(output_dir, "semantic_encoder.onnx"),
        input_names=["audio"],
        output_names=["semantic_mean"],
        dynamic_axes={
            "audio": {0: "batch", 2: "audio_len"},
            "semantic_mean": {0: "batch", 1: "seq_len"},
        },
    )


def export_diffusion_head(model, output_dir, model_type):
    """Export diffusion head to ONNX."""
    print(f"\n--- Exporting diffusion_head ---")

    if model_type == "asr":
        print("  Skipping: ASR model has no diffusion head")
        return

    if model_type == "tts_0.5b":
        prediction_head = model.model.prediction_head
    else:
        prediction_head = model.model.prediction_head

    config = prediction_head.config
    latent_size = config.latent_size  # 64
    hidden_size = config.hidden_size  # 1536 or 896

    wrapper = DiffusionHeadWrapper(prediction_head)
    wrapper.eval()

    dummy_noisy = torch.randn(1, latent_size)
    dummy_timesteps = torch.tensor([500], dtype=torch.long)
    dummy_condition = torch.randn(1, hidden_size)

    export_onnx_model(
        wrapper, (dummy_noisy, dummy_timesteps, dummy_condition),
        os.path.join(output_dir, "diffusion_head.onnx"),
        input_names=["noisy_images", "timesteps", "condition"],
        output_names=["v_prediction"],
        dynamic_axes={
            "noisy_images": {0: "batch"},
            "timesteps": {0: "batch"},
            "condition": {0: "batch"},
            "v_prediction": {0: "batch"},
        },
    )


def export_language_model_full(model, output_dir, model_type, config):
    """Export language model (non-split) for tts_1.5b and asr."""
    print(f"\n--- Exporting language_model (prefill + decode) ---")

    lm_config = config["decoder_config"]
    hidden_size = lm_config["hidden_size"]
    num_layers = lm_config["num_hidden_layers"]
    num_kv_heads = lm_config["num_key_value_heads"]
    num_attn_heads = lm_config["num_attention_heads"]
    head_dim = hidden_size // num_attn_heads
    vocab_size = lm_config["vocab_size"]

    language_model = model.model.language_model
    lm_head = model.lm_head

    print(f"  LM config: H={hidden_size}, L={num_layers}, KV_heads={num_kv_heads}, "
          f"head_dim={head_dim}, V={vocab_size}")

    # --- Prefill ---
    print(f"\n  Exporting prefill...")
    prefill_wrapper = LMPrefillWrapper(language_model, lm_head)
    prefill_wrapper.eval()

    seq_len = 16
    dummy_embeds = torch.randn(1, seq_len, hidden_size)
    dummy_pos_ids = torch.arange(seq_len).unsqueeze(0)

    kv_out_names = get_present_kv_names(num_layers)

    export_onnx_model(
        prefill_wrapper,
        (dummy_embeds, dummy_pos_ids),
        os.path.join(output_dir, "language_model_prefill.onnx"),
        input_names=["inputs_embeds", "position_ids"],
        output_names=["logits"] + kv_out_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 2: "seq_len"} for name in kv_out_names},
        },
    )

    # --- Decode ---
    print(f"\n  Exporting decode...")
    decode_wrapper = LMDecodeWrapper(language_model, lm_head, num_layers, num_kv_heads, head_dim)
    decode_wrapper.eval()

    past_seq_len = 16
    dummy_embeds_1 = torch.randn(1, 1, hidden_size)
    dummy_pos_ids_1 = torch.tensor([[past_seq_len]])

    # Create dummy past KV cache
    dummy_past_kv = []
    for _ in range(num_layers):
        dummy_past_kv.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))  # key
        dummy_past_kv.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))  # value

    kv_in_names = get_lm_kv_names(num_layers)
    kv_present_names = get_present_kv_names(num_layers)

    decode_inputs = (dummy_embeds_1, dummy_pos_ids_1) + tuple(dummy_past_kv)

    export_onnx_model(
        decode_wrapper,
        decode_inputs,
        os.path.join(output_dir, "language_model_decode.onnx"),
        input_names=["inputs_embeds", "position_ids"] + kv_in_names,
        output_names=["logits"] + kv_present_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch"},
            "position_ids": {0: "batch"},
            "logits": {0: "batch"},
            **{name: {0: "batch", 2: "past_seq_len"} for name in kv_in_names},
            **{name: {0: "batch", 2: "total_seq_len"} for name in kv_present_names},
        },
    )


def export_language_model_split(model, output_dir, config):
    """Export split language model for 0.5B streaming: base_lm (4 layers) + tts_lm (20 layers).

    The 0.5B streaming model uses Qwen2Model (no lm_head), so outputs are
    hidden_states [B, S, H] instead of logits [B, S, V].
    """
    print(f"\n--- Exporting split language model (base_lm + tts_lm) ---")

    lm_config = config["decoder_config"]
    hidden_size = lm_config["hidden_size"]
    total_layers = lm_config["num_hidden_layers"]
    tts_layers = config.get("tts_backbone_num_hidden_layers", 20)
    base_layers = total_layers - tts_layers
    num_kv_heads = lm_config["num_key_value_heads"]
    num_attn_heads = lm_config["num_attention_heads"]
    head_dim = hidden_size // num_attn_heads

    print(f"  Split: base_lm={base_layers}L, tts_lm={tts_layers}L")
    print(f"  H={hidden_size}, KV_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  Output: hidden_states (no lm_head)")

    base_lm = model.model.language_model
    tts_lm = model.model.tts_language_model

    # --- base_lm prefill ---
    print(f"\n  Exporting base_lm_prefill ({base_layers} layers)...")
    base_prefill = LMHiddenPrefillWrapper(base_lm)
    base_prefill.eval()

    seq_len = 16
    dummy_embeds = torch.randn(1, seq_len, hidden_size)
    dummy_pos_ids = torch.arange(seq_len).unsqueeze(0)

    kv_out_names = get_present_kv_names(base_layers)

    export_onnx_model(
        base_prefill,
        (dummy_embeds, dummy_pos_ids),
        os.path.join(output_dir, "base_lm_prefill.onnx"),
        input_names=["inputs_embeds", "position_ids"],
        output_names=["hidden_states"] + kv_out_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 2: "seq_len"} for name in kv_out_names},
        },
    )

    # --- base_lm decode ---
    print(f"\n  Exporting base_lm_decode ({base_layers} layers)...")
    base_decode = LMHiddenDecodeWrapper(base_lm, base_layers, num_kv_heads, head_dim)
    base_decode.eval()

    past_seq_len = 16
    dummy_embeds_1 = torch.randn(1, 1, hidden_size)
    dummy_pos_ids_1 = torch.tensor([[past_seq_len]])
    dummy_past_kv = []
    for _ in range(base_layers):
        dummy_past_kv.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))
        dummy_past_kv.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))

    kv_in_names = get_lm_kv_names(base_layers)
    kv_present_names = get_present_kv_names(base_layers)

    export_onnx_model(
        base_decode,
        (dummy_embeds_1, dummy_pos_ids_1) + tuple(dummy_past_kv),
        os.path.join(output_dir, "base_lm_decode.onnx"),
        input_names=["inputs_embeds", "position_ids"] + kv_in_names,
        output_names=["hidden_states"] + kv_present_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch"},
            "position_ids": {0: "batch"},
            "hidden_states": {0: "batch"},
            **{name: {0: "batch", 2: "past_seq_len"} for name in kv_in_names},
            **{name: {0: "batch", 2: "total_seq_len"} for name in kv_present_names},
        },
    )

    # --- tts_lm prefill ---
    print(f"\n  Exporting tts_lm_prefill ({tts_layers} layers)...")
    tts_prefill = LMHiddenPrefillWrapper(tts_lm)
    tts_prefill.eval()

    kv_out_names = get_present_kv_names(tts_layers)

    export_onnx_model(
        tts_prefill,
        (dummy_embeds, dummy_pos_ids),
        os.path.join(output_dir, "tts_lm_prefill.onnx"),
        input_names=["inputs_embeds", "position_ids"],
        output_names=["hidden_states"] + kv_out_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch", 1: "seq_len"},
            "position_ids": {0: "batch", 1: "seq_len"},
            "hidden_states": {0: "batch", 1: "seq_len"},
            **{name: {0: "batch", 2: "seq_len"} for name in kv_out_names},
        },
    )

    # --- tts_lm decode ---
    print(f"\n  Exporting tts_lm_decode ({tts_layers} layers)...")
    tts_decode = LMHiddenDecodeWrapper(tts_lm, tts_layers, num_kv_heads, head_dim)
    tts_decode.eval()

    dummy_past_kv_tts = []
    for _ in range(tts_layers):
        dummy_past_kv_tts.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))
        dummy_past_kv_tts.append(torch.randn(1, num_kv_heads, past_seq_len, head_dim))

    kv_in_names = get_lm_kv_names(tts_layers)
    kv_present_names = get_present_kv_names(tts_layers)

    export_onnx_model(
        tts_decode,
        (dummy_embeds_1, dummy_pos_ids_1) + tuple(dummy_past_kv_tts),
        os.path.join(output_dir, "tts_lm_decode.onnx"),
        input_names=["inputs_embeds", "position_ids"] + kv_in_names,
        output_names=["hidden_states"] + kv_present_names,
        dynamic_axes={
            "inputs_embeds": {0: "batch"},
            "position_ids": {0: "batch"},
            "hidden_states": {0: "batch"},
            **{name: {0: "batch", 2: "past_seq_len"} for name in kv_in_names},
            **{name: {0: "batch", 2: "total_seq_len"} for name in kv_present_names},
        },
    )


def export_connector_weights(model, output_dir, model_type):
    """Extract and save connector/classifier weights as binary files."""
    print(f"\n--- Saving connector weights ---")

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Acoustic connector (all model types)
    save_connector_weights(
        model.model.acoustic_connector,
        os.path.join(weights_dir, "acoustic_connector.bin"),
        "acoustic_connector",
    )

    # Semantic connector (1.5B and ASR only)
    if model_type in ("tts_1.5b", "asr"):
        save_connector_weights(
            model.model.semantic_connector,
            os.path.join(weights_dir, "semantic_connector.bin"),
            "semantic_connector",
        )

    # Speech scaling/bias factors (1.5B and 0.5B)
    if model_type in ("tts_1.5b", "tts_0.5b"):
        if hasattr(model.model, "speech_scaling_factor"):
            save_scalar_buffer(
                model.model.speech_scaling_factor,
                os.path.join(weights_dir, "speech_scaling_factor.bin"),
                "speech_scaling_factor",
            )
        if hasattr(model.model, "speech_bias_factor"):
            save_scalar_buffer(
                model.model.speech_bias_factor,
                os.path.join(weights_dir, "speech_bias_factor.bin"),
                "speech_bias_factor",
            )

    # 0.5B-specific: tts_input_types and tts_eos_classifier
    if model_type == "tts_0.5b":
        if hasattr(model.model, "tts_input_types"):
            save_embedding_weights(
                model.model.tts_input_types,
                os.path.join(weights_dir, "tts_input_types.bin"),
                "tts_input_types",
            )
        if hasattr(model, "tts_eos_classifier"):
            save_binary_classifier_weights(
                model.tts_eos_classifier,
                os.path.join(weights_dir, "tts_eos_classifier.bin"),
                "tts_eos_classifier",
            )

    # Save embed_tokens weights for all models (needed for token -> embedding lookup)
    embed_tokens = model.model.language_model.embed_tokens
    save_embedding_weights(
        embed_tokens,
        os.path.join(weights_dir, "embed_tokens.bin"),
        "embed_tokens",
    )

    # For 0.5B split model, also save tts_lm embed_tokens if different
    if model_type == "tts_0.5b" and hasattr(model.model, "tts_language_model"):
        tts_embed = model.model.tts_language_model.embed_tokens
        # Check if it's a different embedding (might share weights)
        if tts_embed is not embed_tokens:
            save_embedding_weights(
                tts_embed,
                os.path.join(weights_dir, "tts_embed_tokens.bin"),
                "tts_embed_tokens",
            )


def save_model_metadata(output_dir, model_type, config):
    """Save model metadata as JSON for the C++ server."""
    lm_config = config["decoder_config"]
    metadata = {
        "model_type": model_type,
        "hidden_size": lm_config["hidden_size"],
        "num_hidden_layers": lm_config["num_hidden_layers"],
        "num_attention_heads": lm_config["num_attention_heads"],
        "num_key_value_heads": lm_config["num_key_value_heads"],
        "intermediate_size": lm_config["intermediate_size"],
        "vocab_size": lm_config["vocab_size"],
        "head_dim": lm_config["hidden_size"] // lm_config["num_attention_heads"],
        "rope_theta": lm_config.get("rope_theta", 1000000.0),
        "rms_norm_eps": lm_config.get("rms_norm_eps", 1e-6),
        "acoustic_vae_dim": config.get("acoustic_vae_dim", 64),
        "sample_rate": 24000,
        "hop_length": 3200,
    }

    if "semantic_vae_dim" in config:
        metadata["semantic_vae_dim"] = config["semantic_vae_dim"]

    if "diffusion_head_config" in config:
        diff_cfg = config["diffusion_head_config"]
        metadata["diffusion"] = {
            "hidden_size": diff_cfg["hidden_size"],
            "latent_size": diff_cfg["latent_size"],
            "head_layers": diff_cfg["head_layers"],
            "num_train_timesteps": diff_cfg["ddpm_num_steps"],
            "num_inference_steps": diff_cfg["ddpm_num_inference_steps"],
            "beta_schedule": diff_cfg["ddpm_beta_schedule"],
            "prediction_type": diff_cfg["prediction_type"],
        }

    if model_type == "tts_0.5b":
        metadata["tts_backbone_num_hidden_layers"] = config.get("tts_backbone_num_hidden_layers", 20)
        base_layers = lm_config["num_hidden_layers"] - metadata["tts_backbone_num_hidden_layers"]
        metadata["base_lm_layers"] = base_layers
        metadata["tts_lm_layers"] = metadata["tts_backbone_num_hidden_layers"]

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Saved model_metadata.json -> {meta_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export VibeVoice sub-models to ONNX")
    parser.add_argument(
        "--model-dir", type=str, required=True,
        help="Path to model directory (containing config.json + safetensors)",
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["tts_1.5b", "tts_0.5b", "asr"],
        help="Model type to export",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--vibevoice-lib", type=str, default=None,
        help="Path to VibeVoice Python library (if not in PYTHONPATH). "
             "Can also be set via VIBEVOICE_LIB environment variable.",
    )
    parser.add_argument(
        "--skip-onnx", action="store_true",
        help="Skip ONNX export, only save weights",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        help="Only export specified sub-models (e.g., --only language_model). "
             "Valid: acoustic_encoder, acoustic_decoder, semantic_encoder, "
             "diffusion_head, language_model, weights",
    )
    args = parser.parse_args()

    if args.vibevoice_lib:
        os.environ["VIBEVOICE_LIB"] = args.vibevoice_lib

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, config = load_model(args.model_dir, args.model_type)

    def should_export(name):
        """Check if a sub-model should be exported based on --only flag."""
        if args.skip_onnx and name != "weights":
            return False
        if args.only is None:
            return True
        return name in args.only

    with torch.no_grad():
        if should_export("acoustic_encoder"):
            export_acoustic_encoder(model, args.output_dir, args.model_type)
        if should_export("acoustic_decoder"):
            export_acoustic_decoder(model, args.output_dir, args.model_type)
        if should_export("semantic_encoder"):
            export_semantic_encoder(model, args.output_dir, args.model_type)
        if should_export("diffusion_head"):
            export_diffusion_head(model, args.output_dir, args.model_type)

        if should_export("language_model"):
            if args.model_type == "tts_0.5b":
                export_language_model_split(model, args.output_dir, config)
            else:
                export_language_model_full(model, args.output_dir, args.model_type, config)

        if should_export("weights") or args.only is None:
            export_connector_weights(model, args.output_dir, args.model_type)

        # Save metadata (always)
        save_model_metadata(args.output_dir, args.model_type, config)

    # Summary
    print(f"\n{'='*60}")
    print(f"Export complete: {args.model_type}")
    print(f"Output directory: {args.output_dir}")
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


if __name__ == "__main__":
    main()
