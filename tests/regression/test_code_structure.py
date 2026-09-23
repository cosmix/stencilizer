"""Static checks for code structure and axis-mirrored implementations."""

import ast
import io
import re
import tokenize
from collections.abc import Iterator
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src" / "stencilizer"
CORE_ROOT = SOURCE_ROOT / "core"
SIMILARITY_THRESHOLD = 0.75

NamedNode = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
Position = tuple[int, int]
SourceFile = tuple[Path, str, ast.Module]

_IGNORED_TOKENS = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
}


def _source_files(root: Path = SOURCE_ROOT) -> Iterator[SourceFile]:
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        yield path, source, ast.parse(source, filename=str(path))


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _definitions(
    node: ast.AST,
    parents: tuple[str, ...] = (),
    inside_function: bool = False,
) -> Iterator[tuple[NamedNode, str, bool]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = (*parents, child.name)
            yield child, ".".join(names), inside_function
            nested = inside_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            yield from _definitions(child, names, nested)
        else:
            yield from _definitions(child, parents, inside_function)


def _docstring_expression(node: NamedNode) -> ast.Expr | None:
    if not node.body or not isinstance(node.body[0], ast.Expr):
        return None
    expression = node.body[0]
    if isinstance(expression.value, ast.Constant) and isinstance(expression.value.value, str):
        return expression
    return None


def _effective_lines(node: NamedNode) -> int:
    lines = (node.end_lineno or node.lineno) - node.lineno + 1
    docstring = _docstring_expression(node)
    if docstring is not None:
        lines -= (docstring.end_lineno or docstring.lineno) - docstring.lineno + 1
    return lines


def _docstring_ranges(node: FunctionNode) -> list[tuple[Position, Position]]:
    ranges: list[tuple[Position, Position]] = []
    for descendant in ast.walk(node):
        if not isinstance(descendant, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        expression = _docstring_expression(descendant)
        if expression is not None:
            start = (expression.lineno, expression.col_offset)
            end = (expression.end_lineno or expression.lineno, expression.end_col_offset or 0)
            ranges.append((start, end))
    return ranges


def _normalized_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"horizontal|vertical", "AXIS", name)
    name = re.sub(r"width|height", "EXT", name)
    name = re.sub(r"left|right|top|bottom", "SIDE", name)
    name = re.sub(r"\b[xy]\b", "A", name)
    name = re.sub(r"(^|_)[xy](?=\d|_|$)", r"\1A", name)
    return re.sub(r"\b[hv]\b", "AXIS", name)


def _normalized_tokens(source: str, node: FunctionNode) -> tuple[str, ...]:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise ValueError(f"No source segment for function at line {node.lineno}")
    docstrings = _docstring_ranges(node)
    result: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(segment).readline):
        if token.type in _IGNORED_TOKENS:
            continue
        line = node.lineno + token.start[0] - 1
        column = token.start[1] + (node.col_offset if token.start[0] == 1 else 0)
        position = (line, column)
        if token.type == tokenize.STRING and any(
            start <= position < end for start, end in docstrings
        ):
            continue
        result.append(
            _normalized_name(token.string) if token.type == tokenize.NAME else token.string
        )
    return tuple(result)


def similarity_report() -> list[tuple[float, str, str]]:
    """Return every eligible function pair, ordered by decreasing similarity."""
    functions: list[tuple[str, tuple[str, ...]]] = []
    for path, source, tree in _source_files(CORE_ROOT):
        for node, qualname, inside_function in _definitions(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if inside_function or (node.end_lineno or node.lineno) - node.body[0].lineno + 1 < 15:
                continue
            functions.append((f"{_relative(path)}:{qualname}", _normalized_tokens(source, node)))

    report = [
        (SequenceMatcher(None, left_tokens, right_tokens, autojunk=False).ratio(), left, right)
        for (left, left_tokens), (right, right_tokens) in combinations(functions, 2)
    ]
    return sorted(report, key=lambda row: (-row[0], row[1], row[2]))


def test_module_line_limit() -> None:
    offenders = [
        f"{_relative(path)} = {len(source.splitlines())} lines"
        for path, source, _ in _source_files()
        if len(source.splitlines()) > 400
    ]
    assert not offenders, "Modules over 400 physical lines:\n" + "\n".join(offenders)


def test_function_line_limit() -> None:
    offenders = [
        f"{_relative(path)}:{node.lineno} {qualname} = {_effective_lines(node)}"
        for path, _, tree in _source_files()
        for node, qualname, _ in _definitions(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        if _effective_lines(node) > 50
    ]
    assert not offenders, "Functions over 50 lines:\n" + "\n".join(offenders)


def test_class_line_limit() -> None:
    offenders = [
        f"{_relative(path)}:{node.lineno} {qualname} = {_effective_lines(node)}"
        for path, _, tree in _source_files()
        for node, qualname, _ in _definitions(tree)
        if isinstance(node, ast.ClassDef)
        if _effective_lines(node) > 300
    ]
    assert not offenders, "Classes over 300 lines:\n" + "\n".join(offenders)


def test_no_private_font_access_outside_reader() -> None:
    offenders = [
        f"{_relative(path)}:{node.lineno}"
        for path, _, tree in _source_files()
        if path != SOURCE_ROOT / "io" / "reader.py"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        if node.attr == "_font"
        if not (isinstance(node.value, ast.Name) and node.value.id == "self")
    ]
    assert not offenders, "Private font access outside reader.py:\n" + "\n".join(offenders)


def test_no_axis_mirrored_duplication() -> None:
    offenders = [
        f"{left} <-> {right} {ratio:.3f}"
        for ratio, left, right in similarity_report()
        if ratio >= SIMILARITY_THRESHOLD
    ]
    assert not offenders, "Similar core functions:\n" + "\n".join(offenders)
