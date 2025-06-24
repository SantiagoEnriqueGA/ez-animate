# ez-animate

A high-level, declarative Python package for creating common Matplotlib animations with minimal boilerplate code.


## Table of Contents
- [Project Goals](#project-goals)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Development/Contributing](#developmentcontributing)


## Project Goals

`ez-animate` aims to make it easy for data scientists, analysts, educators, and researchers to create standard Matplotlib animations quickly and with minimal code. It abstracts away the complexity of `FuncAnimation`, state management, and repetitive setup, letting you focus on your data and story.

### Why?
- **Complex Setup:** No need to write custom `init` and `update` functions.
- **State Management:** Simplifies handling data and artist states between frames.
- **Repetitive Code:** Reduces boilerplate for standard animations.

### Who is it for?
- **Primary:** Data scientists & analysts (exploratory analysis, presentations, notebooks)
- **Secondary:** Students, educators, and researchers (learning, teaching, publications)

## Features
- **Simple API:** Create animations with a few lines of code
- **Tested & Linted:** High code quality with `pytest` and `ruff`
- **Documentation:** Built with MkDocs (**TBD**)

## Installation

**Pending PyPI release. For now, install directly from GitHub**
```bash
pip install ez-animate
```


## Quickstart

```python
TBD
```

## Development/Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for full development and contributing guidelines.


## Project Structure

```
ez-animate/
├─ .github/
│  ├─ ISSUE_TEMPLATE
│  └─ workflows
├─ examples/
│  ├─ plots
│  ├─ sega_learn
│  └─ sklearn
├─ src/
│  └─ ez_animate
└─ tests

```

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Dependency Management

- All dependencies are in `pyproject.toml`.
- Core: `[project.dependencies]`
- Other: `[project.optional-dependencies]`
- Use `uv pip install -e .[dev]` to install all dev tools.

## Acknowledgments

- Built with inspiration from the Matplotlib community.
- Thanks to all contributors!
