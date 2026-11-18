# AGENTS.md


## Coding & Contribution Standards

Agents modifying or contributing code to this repository must adhere to the following conventions:

**CLI Flag & Logging Conventions:**
   - Use `absl.flags` and `absl.app` (`from absl import app, flags, logging`) for script entry points.
   - Use `logging.info(...)` / `logging.error(...)` instead of raw `print()` for non-trivial script output.
   - Always define sensible defaults and support non-interactive execution (e.g., `--show=False` / `--show_mesh=False`).


**Data Paths & Working Directory:**
   - Prefer relative paths or paths configured through flags (`FLAGS.input_image`, `FLAGS.output_dir`).
   - Ensure output directories are created programmatically (`pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)`).



## Linting & CI Testing

CI workflows are configured in `.github/workflows/`:
- **Flake8:** Enforces syntax errors, undefined names, and PEP8 standards.
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
  ```
- **Pylint:**
  ```bash
  pylint $(git ls-files '*.py')
  ```
- **Pytest:** Runs standard test suites where present.
  ```bash
  pytest
  ```

Ensure all newly added or updated Python scripts pass `flake8` checks without introducing syntax issues or undefined symbols.
