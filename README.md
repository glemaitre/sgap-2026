# SGAP 2026

Course on model evaluation and validation with scikit-learn, skrub, and skore.

## Browse and run the course

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/glemaitre/sgap-2026?quickstart=1) A ready environment with the course packages and the notebooks.

[![Course website](./book/images/jupyterbook_badge.svg)](https://glemaitre.github.io/sgap-2026/) Static HTML with precomputed results.

[![Launch JupyterLite](./book/images/jupyterlite_badge.svg)](https://glemaitre.github.io/sgap-2026/jupyterlite/lab) In-browser execution, no install needed.

[Intro slides](slides-intro.pdf) Slides that introduce the course.

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

The hands-on agent exercise is in the separate `sgap-2026-agentic` repository.

## GitHub Codespaces

Creating a codespace installs the Pixi environment and generates the notebooks. It also installs the Python, Jupyter, and Live Preview extensions, so an HTML file has **Show Preview** on right-click. This README opens in the editor.
