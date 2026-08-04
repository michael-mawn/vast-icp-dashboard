"""
Checks the ground-truth firewall (CLAUDE.md: "the classifier must
rediscover [archetype labels] independently") as an import-graph fact,
not a promise. Plain assert script — no pytest dependency added without
asking; run with `python -m tests.test_firewall`.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _imported_modules(py_file: pathlib.Path) -> set[str]:
    tree = ast.parse(py_file.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _transitive_imports(entry: str, visited: set[str] | None = None) -> set[str]:
    """Follow local (repo) imports transitively from a dotted module path."""
    if visited is None:
        visited = set()
    if entry in visited:
        return visited
    visited.add(entry)
    path = REPO_ROOT / (entry.replace(".", "/") + ".py")
    if not path.exists():
        return visited
    for mod in _imported_modules(path):
        if mod.startswith("config.") or mod.startswith("src.") or mod in ("config", "src"):
            _transitive_imports(mod, visited)
    return visited


def test_classifier_never_imports_generator():
    imports = _transitive_imports("src.classify")
    assert "config.generator" not in imports, (
        f"src.classify has a transitive import path to config.generator: {imports}. "
        "This breaks the ground-truth firewall — the classifier must never see "
        "generation parameters, only config.archetypes."
    )


def test_classifier_never_imports_generate_module():
    imports = _transitive_imports("src.classify")
    assert "src.generate" not in imports, (
        f"src.classify has a transitive import path to src.generate: {imports}. "
        "Same firewall violation as above, via the generation code path."
    )


def test_classifier_only_reads_weekly_profile_table():
    """Checks actual code (string literals, names) for a 'ground_truth'
    reference — ignores comments and docstrings, which may mention it in
    prose (e.g. explaining why the firewall exists)."""
    path = REPO_ROOT / "src/classify.py"
    tree = ast.parse(path.read_text())

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstring_nodes.add(id(body[0].value))

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes:
            if "ground_truth" in node.value:
                offenders.append(node.value)
        if isinstance(node, ast.Name) and "ground_truth" in node.id:
            offenders.append(node.id)

    assert not offenders, (
        f"src/classify.py has executable code referencing 'ground_truth': {offenders}. "
        "The classifier must never read ground-truth labels at inference time."
    )


if __name__ == "__main__":
    test_classifier_never_imports_generator()
    test_classifier_never_imports_generate_module()
    test_classifier_only_reads_weekly_profile_table()
    print("PASS: firewall intact — src.classify has no import path to config.generator, "
          "src.generate, or the ground_truth table.")
