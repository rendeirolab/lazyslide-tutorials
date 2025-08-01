# LazySlide Tutorials

This is a repo for automatically build the LazySlide tutorials notebooks.

## Setup environments

```bash
uv sync

uv run scripts/setup_env.py
```

## To execute one notebook

```bash
uv run jupytext --execute --update path_to.ipynb
```

## To add to cache and execute

```bash
uv run jcache add tutorials/*.ipynb
uv run jcache project execute --executor local-parallel --timeout 1800
```

## Invalid cache

```bash
uv run jcache notebook clear --force
```