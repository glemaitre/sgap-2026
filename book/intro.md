# SGAP 2026

This course introduces model evaluation and validation with scikit-learn, skrub, and skore.

## Notebooks

| | |
| --- | --- |
| [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/glemaitre/sgap-2026?quickstart=1) | A ready environment with the course packages and the notebooks. |
| [![Launch JupyterLite](/images/jupyterlite_badge.svg)](https://glemaitre.github.io/sgap-2026/jupyterlite/lab) | In-browser execution, no install needed. |
| [![Intro slides](/images/pdf_badge.svg)](slides-intro.pdf) | Slides that introduce the course. |

```{tableofcontents}
```

## Run the notebooks locally

The notebooks are in `content/notebooks`. Their sources are in `content/python_files`. Regenerate the notebooks with:

```bash
python utils/convert_to_notebooks.py
```

In JupyterLab you can also open a Python file as a notebook: right click the file and select **Open With**, then **Notebook**.

### Pixi

Install Pixi from the [official website](https://pixi.sh/latest/#installation). This course pins Pixi 0.83.0 for the Codespace and for GitHub Actions.

```bash
pixi run jupyter lab
```

### uv

`requirements.txt` does not install Python. Use Python 3.14.7:

```bash
uv venv --python 3.14.7
uv pip install -r requirements.txt
uv run jupyter lab
```

### pip

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
