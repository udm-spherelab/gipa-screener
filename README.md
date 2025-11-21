# Green Infrastructure PArticipatory Screened 

Automated analysis of research abstracts to identify participatory
methods in urban green infrastructure interventions.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

Run the analysis pipeline:

```bash
uv run python main.py
```

This will:

- Load examples from `in/exemples.json`
- Process abstracts from `in/abstracts.csv`
- Save results to `out/evaluated_by_<model>-<timestamp>.csv`
- Create checkpoints for each analyzed abstract to `out/checkpoint.json`

## Checkpoints

The `main()` function create checkpoints in case of failure
when interacting with OpenAI Rest API. Therefore, if the `main()`
function is launch again, it will start were it failed, avoiding to
rerun the screening process on already anlysed abstract.

When the analysis is performed on all abstracts, the checkpoints file is
removed.

**WARNING**:
Current checkpoints should be removed if we change the
`in/abstracts.csv` file so that their is no mixe results!

## Project Structure

```bash
├── main.py              # Entry point
├── src/
│   ├── analyse.py       # Core analysis functions
│   ├── data.py          # Data loading/saving
│   └── prompt.py        # Prompt generation
├── in/                  # Input data
├── out/                 # Output data and checkpoints
└── dev/                 # Development files
```
