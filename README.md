# SGAP 2026

Course on model evaluation and validation with scikit-learn and skrub, plus an agentic machine learning session.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/glemaitre/sgap-2026?quickstart=1)

## Browse the contents online

Static HTML with precomputed results, and an in-browser JupyterLite lab, are published from this repository with GitHub Pages.

- [Course website](https://glemaitre.github.io/sgap-2026/)
- [JupyterLite](https://glemaitre.github.io/sgap-2026/jupyterlite/lab)

## Run the notebooks

The lesson sources are jupytext percent-format scripts in `content/python_files`. Generate notebooks into `content/notebooks` with:

```bash
python utils/convert_to_notebooks.py
```

In JupyterLab you can also open a script directly: right click the file and select **Open With**, then **Notebook**.

Dataset paths are relative to `content/notebooks` and `content/python_files` (`../datasets/...`).

### Pixi

Pixi installs Python 3.14.7 and the course packages, and it is what builds the website. Install it from the [official website](https://pixi.sh/latest/#installation). This repository pins Pixi 0.83.0 in GitHub Actions and in the Codespace image.

```bash
pixi run jupyter lab
```

Build the static book and JupyterLite:

```bash
pixi run build-book
pixi run build-jupyterlite
pixi run serve-jupyterlite
```

Then open http://[::]:8000/ in a web browser. Firefox often complains about service workers when JupyterLite is served from localhost, which blocks the local dataset files. Use another browser to test that local deployment.

### uv

`requirements.txt` does not install Python. Use Python 3.14.7:

```bash
uv venv --python 3.14.7
uv pip install -r requirements.txt
uv run jupyter lab
```

### pip

Create a virtual environment with Python 3.14.7, then install the same requirements file:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

### conda

```bash
conda env create -f environment.yml
conda activate sgap-2026
jupyter lab
```

Update an existing environment with `conda env update -f environment.yml`.

## Corrections

Solution scripts live in `corrections/python_files` and are not part of the published book or JupyterLite. Generate them with:

```bash
pixi run convert-corrections
```

or:

```bash
python utils/convert_to_notebooks.py --source corrections/python_files --output corrections/notebooks
```

## Agentic machine learning

The hands-on agent session is in [`sessions/agentic-ml`](sessions/agentic-ml). It uses the tephra tables in `data/` and the Pi harness. A Codespace already installs Pi, the `pi-skore` extension, and the skore skills. Instructions for a local harness are in that session's README.

## GitHub Codespaces

Creating a codespace installs the Pixi environment, generates the notebooks, and installs the agentic-session skills. The agentic session README opens in the editor. From `sessions/agentic-ml`, save the shared lecture key with `./configure-key.sh`, then run `pi`.
