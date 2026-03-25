"""
Phase 3: Adaptive Early Stopping Solver Wrapper

Wraps a CEM solver and exits the optimization loop early when the
cost improvement between iterations drops below a threshold (epsilon).

Does NOT modify stable-worldmodel internals — works by overriding the
solve() method while delegating all other protocol methods.

Usage:
    from harness.adaptive_solver import AdaptiveCEMSolver

    base_solver = hydra.utils.instantiate(cfg.solver, model=model)
    solver = AdaptiveCEMSolver(base_solver, epsilon=0.01, min_steps=3)
"""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch


class AdaptiveCEMSolver:
    """CEM solver wrapper with adaptive early stopping.

    Monitors per-iteration best-elite cost. Stops when:
        |cost[i] - cost[i-1]| / |cost[i-1]| < epsilon
    for `patience` consecutive iterations, after at least `min_steps` iterations.

    Attributes:
        stats: Dict tracking per-solve iteration counts for analysis.
    """

    def __init__(
        self,
        solver,
        epsilon: float = 0.01,
        min_steps: int = 3,
        patience: int = 1,
    ):
        """
        Args:
            solver: Base CEM solver instance (already instantiated).
            epsilon: Relative improvement threshold. Stop when improvement < epsilon.
            min_steps: Minimum iterations before early stopping can trigger.
            patience: Number of consecutive below-epsilon iterations before stopping.
        """
        self._solver = solver
        self.epsilon = epsilon
        self.min_steps = min_steps
        self.patience = patience

        # Track statistics across solves for analysis
        self.stats = {
            "iterations_used": [],    # actual iterations per solve call
            "max_iterations": [],     # configured n_steps
            "early_stopped": [],      # whether early stopping triggered
        }

    # ─── Protocol delegation ──────────────────────────────────────

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._solver.configure(action_space=action_space, n_envs=n_envs, config=config)

    @property
    def action_dim(self) -> int:
        return self._solver.action_dim

    @property
    def n_envs(self) -> int:
        return self._solver.n_envs

    @property
    def horizon(self) -> int:
        return self._solver.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    # ─── Adaptive solve ───────────────────────────────────────────

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """CEM solve with adaptive early stopping."""
        start_time = time.time()
        s = self._solver

        outputs = {"costs": [], "mean": [], "var": []}

        mean, var = s.init_action_distrib(init_action)
        mean = mean.to(s.device)
        var = var.to(s.device)

        for start_idx in range(0, s.n_envs, s.batch_size):
            end_idx = min(start_idx + s.batch_size, s.n_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand info dict
            expanded_infos = {}
            for k, v in info_dict.items():
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    v_batch = v_batch.unsqueeze(1)
                    v_batch = v_batch.expand(
                        current_bs, s.num_samples, *v_batch.shape[2:]
                    )
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(
                        v_batch[:, None, ...], s.num_samples, axis=1
                    )
                expanded_infos[k] = v_batch

            final_batch_cost = None
            prev_best_cost = None
            converged_count = 0
            actual_steps = 0

            for step in range(s.n_steps):
                actual_steps = step + 1

                # Sample candidates
                candidates = torch.randn(
                    current_bs,
                    s.num_samples,
                    s.horizon,
                    s.action_dim,
                    generator=s.torch_gen,
                    device=s.device,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                current_info = expanded_infos.copy()
                costs = s.model.get_cost(current_info, candidates)

                topk_vals, topk_inds = torch.topk(
                    costs, k=s.topk, dim=1, largest=False
                )

                batch_indices = torch.arange(
                    current_bs, device=s.device
                ).unsqueeze(1).expand(-1, s.topk)
                topk_candidates = candidates[batch_indices, topk_inds]

                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

                # ─── Early stopping check ─────────────────────
                best_cost = topk_vals[:, 0].mean().item()

                if step >= self.min_steps and prev_best_cost is not None:
                    if abs(prev_best_cost) > 1e-10:
                        rel_improvement = abs(best_cost - prev_best_cost) / abs(prev_best_cost)
                    else:
                        rel_improvement = abs(best_cost - prev_best_cost)

                    if rel_improvement < self.epsilon:
                        converged_count += 1
                        if converged_count >= self.patience:
                            break
                    else:
                        converged_count = 0

                prev_best_cost = best_cost

            # Record stats
            self.stats["iterations_used"].append(actual_steps)
            self.stats["max_iterations"].append(s.n_steps)
            self.stats["early_stopped"].append(actual_steps < s.n_steps)

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        return outputs

    def get_summary(self) -> dict:
        """Return summary statistics of adaptive stopping behavior."""
        if not self.stats["iterations_used"]:
            return {}

        iters = np.array(self.stats["iterations_used"])
        max_iters = self.stats["max_iterations"][0] if self.stats["max_iterations"] else 0
        early = np.array(self.stats["early_stopped"])

        return {
            "total_solves": len(iters),
            "max_iterations": max_iters,
            "mean_iterations": float(np.mean(iters)),
            "median_iterations": float(np.median(iters)),
            "p95_iterations": float(np.percentile(iters, 95)),
            "early_stop_rate": float(np.mean(early)),
            "iteration_reduction": float(1.0 - np.mean(iters) / max_iters) if max_iters > 0 else 0,
        }
