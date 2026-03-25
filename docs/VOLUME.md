# Network Volume Layout

Everything is on the network volume (`/workspace/data/`). Here's what persists across pod sessions:

```
/workspace/data/                              # Network volume (persistent)
├── pusht_expert_train.h5                     # 44GB dataset
├── value_train_data.pt                       # 25MB value function training data
├── pusht/
│   ├── lejepa_object.ckpt                   # 69MB checkpoint (JEPA model)
│   ├── lejepa_weights.ckpt                  # 69MB checkpoint (weights only)
│   ├── pusht_results.txt                    # LeWM eval metrics
│   └── rollout_0..99.mp4                    # 100 LeWM rollout videos
├── checkpoints/
│   └── value_ensemble.pt                    # 3.3MB trained value function ensemble
└── results/
    ├── phase0_lejepa.log                    # LeWM eval log
    ├── phase0_lejepa_pusht_results.txt      # LeWM metrics
    ├── phase0_random.log                    # Random eval log
    ├── phase0_random_pusht_results.txt      # Random metrics
    ├── phase0_timing.log                    # Timing data
    ├── phase1_fidelity.json                 # Fidelity audit results
    ├── phase1_fidelity.log                  # Fidelity audit log
    ├── phase2_sweep.csv                     # Budget sweep results (all configs)
    ├── phase2_convergence_cem_128x15.json   # Per-iteration cost curves
    ├── phase2_convergence_cem_300x10.json   # Per-iteration cost curves
    ├── phase3_adaptive_eps*.json            # Adaptive stopping results
    ├── phase4_value_function.json           # Value function eval results
    ├── phase7_final.json                    # Final benchmark numbers
    └── random_videos/                       # 50 random rollout videos
```
