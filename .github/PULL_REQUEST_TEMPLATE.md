# **[Type]: Brief Description of Change**

*(**Instructions:** Replace `[Type]` with one of: `Feat`, `Fix`, `Docs`, `Style`, `Refactor`, `Perf`, `Test`, `Build`, `CI`, `Chore`. Replace `Brief Description of Change` with a concise summary of the PR's purpose. Example: `Feat: Add Support Vector Machine Classifier`)*

## Description

*(**Instructions:** Provide a clear and concise description of the changes introduced in this pull request. Explain **what** the change is and **why** it is needed. If it addresses a specific problem or adds a new capability, describe it here.)*

*(Optional: If this is a new feature or significant enhancement, consider adding a "Key Features" section like the example below)*
### Key Features:
*(**Instructions:** Only include this section for significant new features. List the main capabilities or highlights introduced.)*
-   Feature 1: Brief description.
-   Feature 2: Brief description.
-   ...

## Key Changes

*(**Instructions:** Detail the specific changes made. Be precise and group related changes. Use the following categories as needed.)*

1.  **New Files**:
    *(List any new files added, including their paths.)*
    -   `path/to/new/module.py`: Purpose of the new file.
    -   `path/to/new/test_module.py`: Purpose of the new test file.
    -   `examples/path/to/new_example.py`: Purpose of the new example.

2.  **Modified Files**:
    *(List existing files that were significantly modified.)*
    -   `path/to/modified/file.py`: Summary of changes made.
    -   `sega_learn/__init__.py`: Reason for modification (e.g., exposing new module).
    -   `requirements.txt`: Added new dependency X.

3.  **Documentation**:
    *(Describe changes to documentation.)*
    -   Updated `README.md`: Added section on [New Feature].
    -   Added/Updated docstrings in `path/to/module.py`.
    -   Created `docs/path/to/new_guide.md`.

4.  **Tests**:
    *(Describe changes related to testing.)*
    -   Added unit tests in `path/to/new/test_module.py`.
    -   Updated tests in `path/to/existing/test_file.py` to reflect changes.

5.  **Examples**:
    *(Describe changes to example scripts.)*
    -   Added `examples/path/to/new_example.py` demonstrating [New Feature].
    -   Updated `examples/path/to/existing_example.py` to use new parameters.

## How to Test

*(**Instructions:** Provide clear, step-by-step instructions for reviewers to verify the changes. Be specific about commands to run and expected outcomes.)*

1.  **Unit Tests**:
    -   Ensure all unit tests pass. Run the test suite using:
        ```bash
        # Example command, adjust as necessary for your project
        python -m unittest discover tests
        # OR
        pytest
        # OR
        python run_all_tests.py
        ```
    -   *(Optional: Mention specific new test files to pay attention to.)* Check `tests/path/to/new_test_module.py`.

2.  **Example Scripts / Manual Verification**:
    -   *(Provide steps to run relevant examples or perform manual checks.)*
    -   Run the new example script: `python examples/path/to/new_example.py`.
    -   Verify that the script runs without errors and produces the expected output/plot/result described in [link to expected result or description].
    -   *(If applicable)* Manually test [specific functionality] by [following these steps...].

3.  **Comparison / Validation** (If Applicable):
    -   *(If the PR implements an algorithm or fixes a bug, suggest how to validate it.)*
    -   Compare the output of [Your Model] with [Reference Implementation, e.g., scikit-learn, statsmodels] using [specific dataset or script].
    -   Verify that [Metric, e.g., accuracy, MSE] is within the expected range.

## Potential Impacts & Considerations

*(**Instructions:** Outline any potential side effects, new dependencies, performance implications, limitations, or other important considerations reviewers should be aware of.)*

-   **Dependencies**: Does this PR add new external dependencies? If so, list them and mention if they are optional.
-   **Performance**: Are there any performance impacts (positive or negative)? Could this change be slow under certain conditions?
-   **Breaking Changes**: Does this PR introduce any backward-incompatible changes?
-   **Limitations**: Are there known limitations or edge cases for the new feature/fix?
-   **Security**: Are there any security implications?

## Related Issues

*(**Instructions:** Link to any relevant GitHub issues. Use keywords like `Fixes #123`, `Closes #123`, or `Related to #456`.)*

-   Fixes #[Issue Number]
-   Related to #[Issue Number]

---

*Self-Review Checklist:*
*(**Instructions:** Check these items yourself before requesting a review. This helps ensure quality and saves reviewers time.)*

-   [ ] Code adheres to project style guidelines (e.g., PEP 8).
-   [ ] Code is well-commented, particularly in hard-to-understand areas.
-   [ ] Docstrings added/updated for new/modified functions and classes.
-   [ ] Corresponding changes to the documentation made (README, guides, etc.).
-   [ ] New and existing unit tests pass locally with my changes.
-   [ ] Added sufficient test coverage for the changes.
-   [ ] No linting errors or warnings reported by the linter (e.g., flake8, pylint).
-   [ ] Ran relevant example scripts to ensure they work with the changes.
-   [ ] Considered potential impacts and documented them.
