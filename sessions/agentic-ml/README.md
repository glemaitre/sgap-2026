# Agentic machine learning

Explore the tephra data in `data/` and build a machine learning model.

`data/train.csv` and `data/test.csv` are the tephra tables. In Pi, start with:

```text
Explore the tephra data in data/ and build a machine learning model.
```

A Codespace already has Pi, the `pi-skore` extension, skore, and the skills. Container setup removes any saved OpenRouter login, trusts this repository, and starts Pi on DeepSeek V4.1 Flash 1017.

The lecture key is shared separately. It is not stored in this repository. From this directory, save it and start Pi:

```bash
./configure-key.sh
set -a && source .env && set +a
pi
```

`./configure-key.sh` writes `.env`. A new Codespace terminal loads that file on its own.

## Install the skills yourself

`skore skills install` opens a wizard. Choose:

- Source: leave `probabl-ai/skore-skills`.
- Skills: all of them. **ML Experimentation Companion** is the 26-skill set. The current release also installs two deprecated compatibility skills, `organize-ml-workspace` and `data-science-python-stack`, so `all` reports 28.
- Agent: **Pi** when you use Pi. For another harness, choose that harness instead.
- Scope: this project directory, not a user-wide install.

The same install without the wizard, from this directory:

```bash
skore skills install all --agent pi
```

That writes the skills to `.agents/skills`, which Pi loads. `skore skills list` should show the full set from the current release.

## On your own computer

Use the course environment so `skore` is installed (Pixi, uv, pip, or conda, from the repository root). Then work in this directory.

### Pi

Install Pi from any directory, then run the remaining commands from this directory:

```bash
curl -fsSL https://pi.dev/install.sh | sh
pi install npm:@probabl/pi-skore
./configure-key.sh
set -a && source .env && set +a
skore skills install all --agent pi
pi
```

`./configure-key.sh` saves the shared lecture key in `.env`. Pi reads `OPENROUTER_API_KEY` and does not ask you to log in. It starts on DeepSeek V4.1 Flash 1017. The first launch asks you to trust this folder before it loads `.pi/settings.json`.

### Another harness

Cursor, Claude Code, Codex, Copilot, and the other harnesses `skore skills install` lists use your own model login. The OpenRouter key is for Pi.

From this directory, run `skore skills install` and choose:

- Source: `probabl-ai/skore-skills`.
- Skills: all of them (**ML Experimentation Companion**, plus the two deprecated compatibility skills in the current release).
- Agent: the harness you will run.
- Scope: this project directory.

Then start that harness here and give it the same task: explore `data/` and build a machine learning model.
