"""
Phase 5: Compiled Inference Wrappers

Applies torch.compile and buffer pre-allocation to the JEPA model
components for reduced planning latency.

Usage:
    from harness.compiled_inference import optimize_model

    model = swm.policy.AutoCostModel(policy_name)
    model = model.to("cuda").eval()
    model = optimize_model(model)  # returns same model with compiled internals
"""

import torch
import torch.nn as nn
from einops import rearrange


def optimize_model(model, compile_predictor=True, compile_encoder=True,
                   backend="inductor", mode="reduce-overhead"):
    """Apply torch.compile to model components for faster inference.

    Modifies the model in-place and returns it. Uses 'reduce-overhead' mode
    by default which enables CUDA graphs for ~3.6x predictor speedup.

    Args:
        model: JEPA model instance (from AutoCostModel).
        compile_predictor: Whether to compile the ARPredictor.
        compile_encoder: Whether to compile the ViT encoder.
        backend: torch.compile backend ('inductor', 'cudagraphs', etc.)
        mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
    """
    torch.set_float32_matmul_precision("high")
    model.eval()
    model.requires_grad_(False)

    if compile_predictor:
        print(f"Compiling predictor with backend='{backend}', mode='{mode}'...")
        model.predictor = torch.compile(model.predictor, backend=backend, mode=mode)

    if compile_encoder:
        print(f"Compiling encoder with backend='{backend}', mode='{mode}'...")
        model.encoder = torch.compile(model.encoder, backend=backend, mode=mode)

    # Replace rollout with buffer-optimized version
    _patch_rollout_with_buffers(model)

    return model


def _patch_rollout_with_buffers(model):
    """Replace the rollout method with a buffer-pre-allocated version.

    The original rollout uses torch.cat on every step, allocating new tensors.
    This version pre-allocates the full buffer and writes in-place.
    """
    original_rollout = model.rollout

    @torch.inference_mode()
    def optimized_rollout(info, action_sequence, history_size=3):
        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # Encode initial info
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = model.encode(_init)
        emb_init = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        info["emb"] = emb_init

        # Flatten batch and sample dimensions
        emb = rearrange(emb_init, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")

        BS = B * S
        D = emb.shape[-1]
        total_steps = H + n_steps + 1  # initial + rollout + final predict

        # Pre-allocate buffers
        emb_buffer = torch.empty(BS, total_steps, D, device=emb.device, dtype=emb.dtype)
        emb_buffer[:, :H, :] = emb

        act_dim = act.shape[-1]
        act_buffer = torch.empty(BS, total_steps - 1, act_dim, device=act.device, dtype=act.dtype)
        act_buffer[:, :H, :] = act

        HS = history_size
        write_pos = H

        for t in range(n_steps):
            # Action encoder on current action buffer
            start = max(0, write_pos - HS)
            act_emb = model.action_encoder(act_buffer[:, start:write_pos, :])
            emb_trunc = emb_buffer[:, start:write_pos, :]
            act_trunc = act_emb

            pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb_buffer[:, write_pos:write_pos + 1, :] = pred_emb

            # Write next action
            act_buffer[:, write_pos:write_pos + 1, :] = act_future_flat[:, t:t + 1, :]
            write_pos += 1

        # Final prediction
        start = max(0, write_pos - HS)
        act_emb = model.action_encoder(act_buffer[:, start:write_pos, :])
        emb_trunc = emb_buffer[:, start:write_pos, :]
        act_trunc = act_emb
        pred_emb = model.predict(emb_trunc, act_trunc)[:, -1:]
        emb_buffer[:, write_pos:write_pos + 1, :] = pred_emb
        write_pos += 1

        # Unflatten
        pred_rollout = rearrange(
            emb_buffer[:, :write_pos, :],
            "(b s) t d -> b s t d", b=B, s=S
        )
        info["predicted_emb"] = pred_rollout

        return info

    model.rollout = optimized_rollout
    print("Patched rollout with pre-allocated buffers.")
