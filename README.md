# Code and data for the submission, "CRAFT: Grounded Multi-Agent Coordination Under Partial Information" 

> **CRAFT** is a multi-agent benchmark for evaluating pragmatic communication 
> in large language models under strict partial information. Three director 
> agents with complementary but incomplete views of a 3D target structure must 
> coordinate through natural language to guide a builder agent toward the correct 
> configuration — a task no single agent can solve alone.

## Supported Models

**Frontier (API):** GPT-4o, GPT-4o-Mini, GPT-4.1-Mini, Claude-Sonnet-4.6, 
Gemini-2.5-Flash, Gemini-3-Flash, Gemini-3.1-Flash-Lite

**Open-weight (local):** Qwen-2.5 7B/14B/32B/72B, Llama-3-8B, Mistral-7B, 
Gemma-2-9B, DeepSeek-V2-Lite


## Repository Structure
```
CRAFT/
├── run_craft.py                      # Main experiment runner (API + local models, CLI)
├── agents/
│   ├── director_agent.py             # DirectorAgent — prompt generation, API/local inference
│   ├── builder_agent.py              # BuilderAgent — move generation, tool use, simulation
│   ├── environment.py                # EnhancedGameState — physics engine, move execution
│   ├── oracle.py                     # Oracle move enumeration and validation
│   ├── builder_tools.py              # simulate_move — dry-run move validation
│   └── common_ground_agent.py        # Optional common ground tracking agent
├── structure_generator_v2.py         # Procedural 3D structure generation
├── task_progress_tracker.py          # Per-turn progress metrics
├── local_model_utils.py              # HuggingFace model loading utilities
├── judge_pragmatics.py               # PS-Judge implementation
├── sg_mm_judge_calls.py              # SG-Judge and MM-Judge implementation
├── plot_failure_taxonomy.py          # Failure mode analysis and plotting
├── train_sft.py                      # SFT training for director models (TRL + LoRA) (optional, not used in paper)
├── train_dpo.py                      # DPO training for director models (TRL + LoRA) (optional, not used in paper)
├── test_game_state_tracking.py       # Unit tests — physics engine correctness
├── test_oracle.py                    # Oracle validation suite — move stats and coverage
├── run_craft_all.sh                    # Full experimental run for the paper for generating game logs
├── data/
│   └── structures_dataset_20.json    # 20 evaluation structures (7 simple,
│                                     # 8 medium, 5 complex; 21-25 blocks)
└── plotting_scripts/                 # Analysis and visualization scripts
```
## Installation
```bash
git clone https://github.com/csu-signal/CRAFT
cd CRAFT
pip install -r requirements.txt
```

Set API keys before running frontier model experiments:
```bash
export OPENAI_API_KEY=...
export CLAUDE_API_KEY=...
export GEMINI_API_KEY=...
```

## Running Experiments

CRAFT uses a single CLI entry point for both API and local models used in the experiments. 

### Full Experiment Run (Paper Settings)
The script runs all 9 frontier API models followed by all 8 open-weight local models with the exact settings used in the paper: oracle candidate moves enabled, builder tool use disabled, 20 turns, all 20 structures.

```bash
chmod +x run_craft_all.sh
./run_craft_all.sh
```

This sequentially runs:
- **API models** (no GPU required): `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4o`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `claude-haiku-4-5`, `claude-sonnet-4-6`
- **Local models** (GPU required): `qwen-7b`, `qwen-14b`, `qwen-32b`, `llama-8b`, `mistral-7b`, `gemma-9b`, `deepseek-v2-lite`, `qwen-72b`

`qwen-32b` and `qwen-72b` are automatically loaded with 4-bit quantization. 

Results are saved under `craft_results/api/` and `craft_results/local/` respectively.

### Running a Single Model
```bash
# API model
python run_craft.py --mode api --director gpt-4o --oracle --oracle_n 5 --no_tools --turns 20

# Local model
python run_craft.py --mode local --director qwen-7b --oracle --oracle_n 5 --no_tools --turns 20
```

### Running a Subset of Structures
```bash
python run_craft.py --mode api --director gpt-4o-mini --structures 0,1,5 --oracle --oracle_n 5 --no_tools --turns 20
```

## Judge Evaluation

After generating game logs with `run_craft_all.sh` or `run_craft.py`, run the three judges to evaluate director communication quality.

### Step 1 — Configure Paths

In `sg_mm_judge_calls.py`, set the paths to your game logs:

```python
ORACLE_DIR_FRONTIER = "craft_results/api"    # frontier model game logs
ORACLE_DIR_BASE     = "craft_results/local"  # open-weight model game logs
BASE_DIR            = Path("judge_results_sg_mm")  # output directory
```

Game logs are expected in the format:
```
craft_results/api/{model_name}_gpt-4o-mini/craft_{structure_id}_{run}.json
craft_results/local/{model_name}_gpt-4o-mini/craft_{structure_id}_{run}.json
```

### Step 2 — Run SG and MM Judges

Evaluates each director's spatial grounding (SG) and mind modeling (MM) quality turn-by-turn:

```bash
python sg_mm_judge_calls.py
```

Results saved to `judge_results_sg_mm/run_{1,2,3}/master.pkl`.

### Step 3 — Run PS Judge

Evaluates whether collective director output was pragmatically sufficient to guide the builder toward an oracle-correct move:

```bash
python judge_pragmatics.py
```

Results saved to `judge_results_ps/run_{1,2,3}/ps_results.pkl`.

### Judge Output Format

Each `master.pkl` contains a dict keyed by model name, with per-turn per-director SG and MM scores:

```python
{
  "gpt-4o": {
    "sg": [{"model": ..., "structure_id": ..., "turn": ..., "director": ...,
             "scores": {"SG1": {"answer": "Yes", "reason": "..."}, ...},
             "sg_score": 0.857}, ...],
    "mm": [{"mm_score": 0.625, ...}, ...]
  }, ...
}
```

Each `ps_results.pkl` contains per-turn PS scores with condition labels (C1\_followed / C2\_not\_followed) and per-question answers (PS1–PS6).

**All CLI options:**
```
--mode        api | local               Director mode (default: api)
--director    model name or key         Single model to run (default: all)
--builder     model name                Builder model (default: gpt-4o-mini)
--dataset     path                      Structure dataset (default: data/structures_dataset_20.json)
--turns       int                       Max turns per game (default: 20)
--run         int                       Run index for seeding (default: 3)
--oracle                                Enable oracle candidate moves
--oracle_n    int                       Oracle moves per turn (default: 5)
--no_tools                              Disable builder simulate_move tool
--structures  e.g. 0,1,5               Subset of structures to run (default: all 20)
--quantize    4bit | 8bit               Quantization for large local models
--output      path                      Custom output directory
```
