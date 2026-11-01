# Agentic machine learning

Explore the tephra data in `data/` and build a machine learning model.

`data/train.csv` and `data/test.csv` are the tephra tables. In Pi, start with:

```text
Explore the tephra data in data/ and build a machine learning model.
```

## Already done in a Codespace

The image installed Pi and the `pi-skore` extension:

```bash
curl -fsSL https://pi.dev/install.sh | sh
pi install npm:@probabl/pi-skore
```

Creating the Codespace then ran, from the repository root:

```bash
bash .devcontainer/prepare-pi.sh
pixi install
pixi run convert-to-notebooks
cd sessions/agentic-ml && pixi run skore skills install all --agent pi
```

`prepare-pi.sh` drops any saved OpenRouter login, trusts this repository, and selects `deepseek/deepseek-v4.1-flash-1017`. `pixi install` and `pixi run convert-to-notebooks` install the course environment and generate the notebooks. `skore skills install all --agent pi` wrote the skills to `.agents/skills`, which Pi loads. **ML Experimentation Companion** is the 26-skill set. The current release also installs two deprecated compatibility skills, `organize-ml-workspace` and `data-science-python-stack`, so `skore skills list` shows 28.

## What is left

The lecture key is shared separately. It is not stored in this repository. From this directory, save it and start Pi:

```bash
./configure-key.sh
set -a && source .env && set +a
pi
```

`./configure-key.sh` writes `.env`. A new Codespace terminal loads that file on its own. Pi reads `OPENROUTER_API_KEY` and does not ask you to log in.

## On your own computer

Use the course environment so `skore` is installed (Pixi, uv, pip, or conda, from the repository root). Then work in this directory. The commands below reach the same stage as the Codespace.

### Pi

Install Pi from any directory, then run the remaining commands from this directory:

```bash
curl -fsSL https://pi.dev/install.sh | sh
pi install npm:@probabl/pi-skore
skore skills install all --agent pi
./configure-key.sh
set -a && source .env && set +a
pi
```

`skore skills install` opens a wizard if you omit `all --agent pi`. Choose source `probabl-ai/skore-skills`, all skills, agent **Pi**, and this project directory. `./configure-key.sh` saves the shared lecture key in `.env`. Pi starts on DeepSeek V4.1 Flash 1017. The first launch asks you to trust this folder before it loads `.pi/settings.json`.

### Another harness

Cursor, Claude Code, Codex, Copilot, and the other harnesses `skore skills install` lists use your own model login. The OpenRouter key is for Pi.

From this directory, run `skore skills install` and choose:

- Source: `probabl-ai/skore-skills`.
- Skills: all of them (**ML Experimentation Companion**, plus the two deprecated compatibility skills in the current release).
- Agent: the harness you will run.
- Scope: this project directory.

Then start that harness here and give it the same task: explore `data/` and build a machine learning model.
