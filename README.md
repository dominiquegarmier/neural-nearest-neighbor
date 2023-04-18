[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/DominiqueGarmier/neural-nearest-neighbor/main.svg)](https://results.pre-commit.ci/latest/github/DominiqueGarmier/neural-nearest-neighbor/main) [![pytest](https://github.com/DominiqueGarmier/neural-nearest-neighbor/actions/workflows/test.yml/badge.svg)](https://github.com/DominiqueGarmier/neural-nearest-neighbor/actions/workflows/test.yml)

# neural $k$ nearest neighbor

Implementation of [neural k nearest neighbor](https://arxiv.org/abs/1810.12575) in pytorch, a differentiable replacement for kNN.

### Installation

```
pip install git+ssh://git@github.com/dominiquegarmier/neural-nearest-neighbor
```

and for development:

```
pip install -r requirements-dev.txt
```

### Notes

When using `value` you have to make some additional assumtions to get the same convergence conditions as outlined in the [paper](https://arxiv.org/abs/1810.12575).
Notably you have to assume that there exists some continous $f: k \mapsto v$ that maps keys to values.

### Citations

```bibtex
@misc{plötz2018neural,
      title={Neural Nearest Neighbors Networks},
      author={Tobias Plötz and Stefan Roth},
      year={2018},
      eprint={1810.12575},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
