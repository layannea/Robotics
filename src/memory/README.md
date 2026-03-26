# Memory Module

Retrieves similar past episodes and injects their code as hints into the LLM prompt, so the robot can reuse solutions from previous runs.

# How it works

After each run, the log file is parsed into an episode record and saved to `episodes.json`. On the next run, the planner and affordance map LMPs query the store for similar past episodes and prepend their code as `# hint:` comments into the prompt.

Similarity is Jaccard over instruction tokens + object names, with `0.1 ≤ sim < 0.99` (the upper bound avoids injecting the exact same instruction back).

# Files

- `memory_store.py` — load, save, and retrieve episodes
- `log_parser.py` — parse run log `.txt` files into episode dicts
- `episodes.json` — the episode database (auto-created on first save)

# Episode format

```json
{
  "task": "PushButton",
  "instruction": "push the rose button",
  "objects": ["button"],
  "planner_code": "composer(\"push the button\")",
  "affordance_maps": [
    {
      "query": "the button",
      "code": "affordance_map = button.occupancy_map\nret_val = affordance_map"
    }
  ],
  "outcome": "success",
  "reward": 1.0
}
```

`outcome` is `"success"`, `"failure"`, or `"stuck"` (no reward line in log).

# Usage

Toggle in `src/LMP.py`:
```python
USE_MEMORY = True   # enable
USE_MEMORY = False  # disable for baseline comparison
```

Rebuild the store from all existing logs:
```bash
cd src
python memory/memory_store.py logs/
```

Successful runs are saved automatically during `playground.ipynb` execution.
