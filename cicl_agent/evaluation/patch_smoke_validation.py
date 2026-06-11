"""Targeted executable checks for real-code patch-generation smoke artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeCheck:
    instance_id: str
    target_path: str
    check_type: str
    passed: bool
    details: dict[str, Any]


def validate_patch_smoke_artifact(root: str | Path, output: str | Path | None = None) -> dict:
    """Run task-specific checks for the n=3 Opus patch smoke artifact."""

    root_path = Path(root)
    checks = [
        _validate_qdp(root_path / "astropy__astropy-14365"),
        _validate_rst(root_path / "astropy__astropy-14182"),
        _validate_fits(root_path / "astropy__astropy-6938"),
    ]
    summary = {
        "artifact_root": str(root_path),
        "n_checks": len(checks),
        "n_passed": sum(1 for check in checks if check.passed),
        "official_swebench_harness": "not run",
        "claim_boundary": (
            "Targeted executable checks for patch-smoke artifacts only; "
            "not an official SWE-bench pass rate."
        ),
        "checks": [
            {
                "instance_id": check.instance_id,
                "target_path": check.target_path,
                "check_type": check.check_type,
                "passed": check.passed,
                "details": check.details,
            }
            for check in checks
        ],
    }
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        _write_csv(output_path.with_suffix(".csv"), checks)
    return summary


def _validate_qdp(task_dir: Path) -> SmokeCheck:
    base = _read_first_existing(task_dir / "base_source.py", task_dir / "prompt_context_source.py")
    patched = _read_first_existing(task_dir / "patched_source.py", task_dir / "normalized_patched_source.py")
    before = _call_qdp_line_type(base, "read serr 1 2")
    after_lower = _call_qdp_line_type(patched, "read serr 1 2")
    after_upper = _call_qdp_line_type(patched, "READ SERR 1 2")
    passed = (not before["ok"]) and after_lower.get("value") == "command" and after_upper.get("value") == "command"
    return SmokeCheck(
        instance_id="astropy__astropy-14365",
        target_path="astropy/io/ascii/qdp.py",
        check_type="executable_function_slice",
        passed=passed,
        details={
            "pre_lowercase": before,
            "post_lowercase": after_lower,
            "post_uppercase": after_upper,
        },
    )


def _validate_rst(task_dir: Path) -> SmokeCheck:
    base = _read_first_existing(task_dir / "base_source.py", task_dir / "prompt_context_source.py")
    patched = _read_first_existing(
        task_dir / "repair_feedback2_normalized_patched_source.py",
        task_dir / "repair_feedback2_patched_source.py",
        task_dir / "patched_source.py",
    )
    before = _exercise_rst(base)
    after = _exercise_rst(patched)
    passed = (
        before["init_with_header_rows"]["ok"] is False
        and after["init_with_header_rows"]["ok"] is True
        and after["read_start_line_after_read"] == 4
        and after["write_border_index_value"] == "unit-border"
    )
    return SmokeCheck(
        instance_id="astropy__astropy-14182",
        target_path="astropy/io/ascii/rst.py",
        check_type="executable_class_slice",
        passed=passed,
        details={"before": before, "after": after},
    )


def _validate_fits(task_dir: Path) -> SmokeCheck:
    patched = _read_first_existing(task_dir / "patched_source.py")
    replacement = _exercise_non_inplace_replace()
    assigns_replace = "output_field[:] = output_field.replace" in patched
    passed = replacement["pre_after_replace"] == "1.0E+00" and replacement["post_after_assignment"] == "1.0D+00" and assigns_replace
    return SmokeCheck(
        instance_id="astropy__astropy-6938",
        target_path="astropy/io/fits/fitsrec.py",
        check_type="executable_replacement_semantics",
        passed=passed,
        details={**replacement, "patched_assigns_replace_result": assigns_replace},
    )


def _call_qdp_line_type(source: str, line: str) -> dict:
    try:
        namespace: dict[str, Any] = {}
        tree = ast.parse(source)
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_line_type"
        )
        module = ast.Module(
            body=[
                ast.Import(names=[ast.alias(name="re")]),
                function,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        exec(compile(module, "<qdp_line_type>", "exec"), namespace)
        return {"ok": True, "value": namespace["_line_type"](line)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def _exercise_rst(source: str) -> dict:
    try:
        namespace = _rst_namespace()
        tree = ast.parse(source)
        class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RST")
        module = ast.Module(body=[class_def], type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module, "<rst_class>", "exec"), namespace)
        cls = namespace["RST"]
        result: dict[str, Any] = {"init_with_header_rows": {"ok": True}}
        try:
            instance = cls(header_rows=["name", "unit"])
        except Exception as exc:  # noqa: BLE001
            return {
                "init_with_header_rows": {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
                "read_start_line_after_read": None,
                "write_border_index_value": None,
            }
        lines = ["top-border", "name-border", "unit-border", "data", "bottom-border"]
        written = instance.write(lines)
        result["write_border_index_value"] = written[0] if written else None
        instance.read(["dummy"])
        result["read_start_line_after_read"] = getattr(instance.data, "start_line", None)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "init_with_header_rows": {
                "ok": False,
                "error": type(exc).__name__,
                "message": str(exc),
            },
            "read_start_line_after_read": None,
            "write_border_index_value": None,
        }


def _rst_namespace() -> dict[str, Any]:
    class FixedWidth:
        def __init__(self, delimiter_pad=None, bookend=False, header_rows=None):
            self.delimiter_pad = delimiter_pad
            self.bookend = bookend
            self.header = type("Header", (), {"header_rows": header_rows or ["name"]})()
            self.data = type("Data", (), {"start_line": 3})()

        def write(self, lines):
            return list(lines)

        def read(self, table):
            return table

    return {
        "FixedWidth": FixedWidth,
        "SimpleRSTData": type("SimpleRSTData", (), {}),
        "SimpleRSTHeader": type("SimpleRSTHeader", (), {}),
    }


def _exercise_non_inplace_replace() -> dict:
    pre = _FakeCharField(b"1.0E+00")
    pre.replace(b"E", b"D")
    post = _FakeCharField(b"1.0E+00")
    post[:] = post.replace(b"E", b"D")
    return {
        "engine": "fake_char_field_with_numpy_chararray_like_replace",
        "pre_after_replace": bytes(pre).decode("ascii"),
        "post_after_assignment": bytes(post).decode("ascii"),
    }


class _FakeCharField:
    """Small chararray-like object where replace returns a copy."""

    def __init__(self, value: bytes) -> None:
        self.value = value

    def replace(self, old: bytes, new: bytes) -> bytes:
        return self.value.replace(old, new)

    def __setitem__(self, key, value: bytes) -> None:
        if key != slice(None, None, None):
            raise TypeError("Only whole-field assignment is supported in this smoke check.")
        self.value = value

    def __bytes__(self) -> bytes:
        return self.value


def _read_first_existing(*paths: Path) -> str:
    for path in paths:
        if path.exists():
            return path.read_text(encoding="utf-8")
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of the expected smoke files exist: {joined}")


def _write_csv(path: Path, checks: list[SmokeCheck]) -> None:
    rows = [
        {
            "instance_id": check.instance_id,
            "target_path": check.target_path,
            "check_type": check.check_type,
            "passed": int(check.passed),
        }
        for check in checks
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate real-code Opus patch-smoke artifacts.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = validate_patch_smoke_artifact(args.root, args.output)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
