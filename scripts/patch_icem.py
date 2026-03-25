#!/usr/bin/env python3
"""
Patch stable_worldmodel ICEMSolver to fix action bounds shape mismatch.

The bug: ICEMSolver.configure sets _action_low/high from action_space.low[0]
which has shape (action_dim,), but the solver's action_dim property includes
action_block multiplication. When action_block > 1, candidates.clamp() fails
because bounds shape (2,) != candidates dim (10 = 2 * action_block).

Run this after `pip install stable-worldmodel`:
    python scripts/patch_icem.py
"""

import re
from pathlib import Path

import stable_worldmodel.solver.icem as icem_module

icem_path = Path(icem_module.__file__)
source = icem_path.read_text()

OLD = """        if isinstance(action_space, Box):
            self._action_low = torch.tensor(action_space.low[0], device=self.device, dtype=torch.float32)
            self._action_high = torch.tensor(action_space.high[0], device=self.device, dtype=torch.float32)"""

NEW = """        if isinstance(action_space, Box):
            raw_low = torch.tensor(action_space.low[0], device=self.device, dtype=torch.float32)
            raw_high = torch.tensor(action_space.high[0], device=self.device, dtype=torch.float32)
            # Repeat bounds to match action_dim (which includes action_block)
            n_repeats = config.action_block if hasattr(config, 'action_block') else 1
            self._action_low = raw_low.repeat(n_repeats)
            self._action_high = raw_high.repeat(n_repeats)"""

if OLD in source:
    source = source.replace(OLD, NEW)
    icem_path.write_text(source)
    print(f"Patched {icem_path}")
elif "raw_low" in source:
    print(f"Already patched: {icem_path}")
else:
    print(f"WARNING: Could not find expected code in {icem_path}")
    print("The iCEM solver may have been updated. Check manually.")
