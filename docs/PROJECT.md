# LeWM Planning Harness: Project Overview

## What Is This

LeWM is a 15M-parameter world model that predicts the future in latent space. It's small, fast to train, and its representations encode real physical structure. But out of the box, it plans slowly (1-2 seconds per decision) and scores trajectories with raw embedding distance — a blunt instrument.

This project wraps LeWM in a **planning harness** that makes it plan better and plan faster, turning it from a research artifact into something that can run onboard a robot and make decisions in real time.

The end state: a system where LeWM imagines candidate futures, a learned reward model picks the best one, and a low-level policy executes it — all running on a Jetson at 10-50 Hz.

## Why This Matters

The robotics field is converging on two approaches to robot intelligence:

1. **VLM-backbone VLAs** (RT-2, OpenVLA, Pi0) — Large vision-language models that map observations directly to actions. They understand semantics but don't model physics. They're slow (5-10 Hz on cloud GPUs) and expensive to deploy.

2. **World-model-based planning** (Dreamer, TD-MPC, LeWM) — Small dynamics models that imagine futures and search for good actions. They understand physics but are limited by their planning loop.

The opportunity: **a world model that plans well enough and fast enough can replace the VLM as the backbone of a VLA**. LeWM is the right starting point — 15M params vs 7-55B for VLM-based VLAs, latent-space planning instead of pixel generation, and a representation that already encodes physical quantities.

What's missing is the infrastructure to make it plan intelligently and deploy at real-time speeds. That's what this harness provides.

## What We're Building

Four layers, each building on the last:

### Layer 1 — Faster Planning
Take the existing CEM planner from 300 samples x 30 iterations (45,000 forward passes, ~1-2s) down to 64 samples x 5 iterations (~320 forward passes, ~50-100ms) without losing task success. This is done through:
- iCEM solver (colored noise, elite retention from prior timestep)
- Adaptive early stopping (exit when cost converges)
- TensorRT compilation of the encoder and predictor
- CUDA Graphs to eliminate kernel launch overhead

### Layer 2 — Smarter Trajectory Scoring
Replace the current cost function (MSE between predicted embedding and goal embedding) with a learned reward model that operates on LeWM's latent trajectories. A 1-5M parameter Transformer or MLP that takes a sequence of 192-dim embeddings and outputs a progress score. Trained offline using expert demonstrations and VLM-generated quality labels. This captures task-relevant structure that raw embedding distance misses — intermediate progress, contact events, orientation constraints.

### Layer 3 — Dual-Mode Execution
A small amortized policy (~2-5M params) distilled from the planner handles routine states at 50+ Hz. The full world model + reward model planning loop activates for novel or high-stakes states at 10-20 Hz. A confidence gate decides which mode to use.

### Layer 4 — Offline Improvement Pipeline
A server-side pipeline that continuously improves all three components:
- Runs large-budget CEM to generate expert (state, action) pairs for policy distillation
- Uses a VLM (Qwen-VL-4B) to label trajectory quality for reward model training
- Fine-tunes the world model predictor on task-specific data

## How We Evaluate Success

### Primary Metrics

**Task success rate** — Does the robot complete the task? Measured on PushT (2D manipulation), TwoRoom (navigation), Cube (3D manipulation), and Reacher (continuous control). The baseline is LeWM's current eval: 50 episodes per task with CEM planning.

**Planning latency (ms/decision)** — How long does a single planning step take? Current baseline: ~1000-2500ms on AGX Orin. Target: <100ms.

**Control frequency (Hz)** — How many decisions per second? Current: <1 Hz. Target: 10-20 Hz in deliberate mode, 50+ Hz in fast mode.

### Secondary Metrics

**Planning efficiency curve** — Success rate as a function of CEM sample budget. Shows whether the reward model lets us achieve the same success with fewer samples.

**Anytime performance** — Success rate if we stop planning after k CEM iterations. Validates adaptive early stopping.

**Reward-task correlation** — Does the learned reward model's score actually predict task success? Measured as Pearson r between trajectory reward and binary outcome.

**Representation utility** — Linear probing accuracy for physical quantities (position, angle, velocity) from LeWM's latent space. Inherited from the original paper; we track it to ensure harness modifications don't degrade the representation.

**Amortization gap** — Success rate of the distilled fast policy vs. the full planner. Measures how much we lose by skipping planning.

### What Success Looks Like

| Milestone | Success Criterion |
|-----------|-------------------|
| M1: Faster planning | Same success rate at 10x fewer samples (64 vs 300) |
| M2: Reward model | Higher success rate than MSE cost at equal sample budget |
| M3: Real-time on Jetson | 10+ Hz deliberate planning on AGX Orin with TensorRT |
| M4: Fast policy | >80% of planner success rate at 50+ Hz |
| M5: Integrated system | Dual-mode harness running on Jetson, completing manipulation tasks end-to-end |

### What Failure Looks Like

- Reducing CEM budget tanks success rate, meaning the world model's predictions aren't good enough for efficient planning — fix the model, not just the planner
- Learned reward model doesn't outperform MSE cost — the latent space is already well-structured and a simpler cost is sufficient
- Amortized policy can't capture the planner's behavior — the action distribution is too multimodal for a feedforward network
- TensorRT export breaks the autoregressive rollout loop — need custom ONNX tracing for the sequential prediction

## Steps

### Phase 1: Validate (Week 1-2)
1. Run baseline eval on PushT with pretrained checkpoint — establish success rate and timing
2. Swap CEM for iCEM — measure success rate at 300, 128, 64, 32 samples
3. Sweep CEM iterations (30, 15, 10, 5) — find the minimum viable budget
4. Implement adaptive early stopping — log cost convergence curves
5. **Gate**: Can we match baseline success at 64 samples / 5 iterations?

### Phase 2: Reward Model (Week 3-4)
1. Collect LeWM latent trajectories from expert demonstrations (run encoder on dataset)
2. Label trajectory quality: use expert progress + VLM scoring
3. Train small reward model on (latent_trajectory, score) pairs
4. Plug reward model into CEM as replacement cost function
5. **Gate**: Does reward model improve success rate over MSE cost?

### Phase 3: Speed (Week 5-6)
1. Export LeWM encoder + predictor to ONNX → TensorRT
2. Handle autoregressive rollout in TRT (custom loop or scripted module)
3. Benchmark on AGX Orin: measure ms/decision end-to-end
4. Apply CUDA Graphs if kernel launch overhead is significant
5. **Gate**: <100ms per deliberate planning step on AGX Orin?

### Phase 4: Policy Distillation (Week 7-8)
1. Run optimized planner on diverse (start, goal) pairs — collect (observation, action) dataset
2. Train feedforward policy network on planner outputs
3. Add uncertainty estimation (ensemble or MC dropout) for mode switching
4. Implement dual-mode harness with confidence gating
5. **Gate**: Fast policy achieves >80% of planner success at 50+ Hz?

### Phase 5: Integration (Week 9-10)
1. Full system on Jetson: camera input → encoder → planner/policy → actuator output
2. End-to-end latency profiling
3. Stress testing: novel objects, perturbed initial states, varied goals
4. Compare against VLM-backbone VLA baselines (OpenVLA, Pi0) on same tasks
5. **Gate**: System completes manipulation tasks end-to-end on real hardware

## Related Work

This project builds directly on:
- **LeWM** (Maes et al., 2026) — The world model at the core. 15M params, JEPA architecture, stable end-to-end training.
- **iCEM** (Pinneri et al., 2020) — Sample-efficient planning with colored noise and elite retention.
- **SimDist** (2025) — Simulation distillation: train reward/value models in sim, freeze for deployment. Small 1-layer Transformer reward models running at 50 Hz.
- **SARM** (2025) — Stage-aware reward modeling with frozen CLIP encoder + 60M Transformer. Shows small reward models work for trajectory scoring.
- **TD-MPC2** (Hansen et al., 2024) — Demonstrates amortized policy + online planning hybrid. Policy prior warm-starts the planner.
- **GR-2** (ByteDance, 2024) — Video-pretrained world model as VLA backbone. 97.7% success on CALVIN, 2x generalization improvement over non-world-model approaches.

## Repository Structure

```
le-wm/
  jepa.py          # LeWM world model (encoder, predictor, rollout, cost)
  module.py        # Transformer blocks, attention, embedders, SIGReg
  train.py         # Training loop
  eval.py          # Evaluation with CEM/Adam planning
  utils.py         # Preprocessing, callbacks
  config/          # Hydra configs for training and eval
  docs/
    PROJECT.md     # This document
    HARNESS.md     # Technical architecture details
```
