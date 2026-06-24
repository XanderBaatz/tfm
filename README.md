# Trivialized Momentum Flow Matching on Lie Groups

## 🏗️ Project Structure

```
.
├── tools/                  # Reusable utility modules
│   ├── config/             # Configuration management (Settings, FastAPI config)
│   ├── logger/             # Logging utilities (Local & Google Cloud formatters)
│   └── tracer/             # Performance tracing (Timer decorator/context manager)
├── tests/                  # Test suite (mirrors tools/ structure)
│   └── tools/              # Unit tests for utility modules
├── docs/                   # MkDocs documentation
│   ├── getting-started/    # Setup guides
│   ├── guides/             # Tool usage guides
│   ├── configurations/     # Configuration references
│   └── usecases/           # Real-world examples
├── .devcontainer/          # Dev Container configuration
├── .github/                # GitHub Actions workflows, PR templates, and review checklists
├── CODE_OF_CONDUCT.md      # Community Code of Conduct
├── CONTRIBUTING.md         # Contribution guidelines
├── CLAUDE.md               # Claude Code development guidance
├── noxfile.py              # Task automation configuration (test, lint, fmt)
├── pyproject.toml          # Project metadata and dependencies (uv)
├── ruff.toml               # Ruff linter/formatter configuration
└── pytest.ini              # Pytest configuration (75% coverage requirement)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This template is built on top of excellent open-source tools:

- **[uv](https://github.com/astral-sh/uv)** by Astral - Ultra-fast Python package manager
- **[Ruff](https://github.com/astral-sh/ruff)** by Astral - Lightning-fast linter and formatter
- **[ty](https://github.com/astral-sh/ty)** by Astral - Static type checker for Python
- **[nox](https://nox.thea.codes/)** - Flexible task automation for Python
- **[pytest](https://pytest.org/)** - Testing framework for Python
- **[MkDocs](https://www.mkdocs.org/)** - Documentation site generator
