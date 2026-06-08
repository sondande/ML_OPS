"""
version_check_agent.py — Rule-based library API version compliance checker.

Parses pinned versions from pyproject.toml, loads version_rules.yaml, and
scans Python source files for patterns that indicate legacy API usage
incompatible with the pinned versions.

Exits with code 1 if any violations are found (blocks CI).
Exits with code 0 if clean.

Usage:
    python version_check_agent.py \\
        --pyproject pyproject.toml \\
        --rules .github/agents/version_rules.yaml \\
        --scan-dir src \\
        --scan-dir app \\
        --output /tmp/version_report.md
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # pip install tomli

import yaml


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_pinned_versions(pyproject_path: Path) -> dict[str, str]:
    """Return {package_name: version_string} from pyproject.toml dependencies."""
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    versions: dict[str, str] = {}
    for dep in deps:
        # Match:  package==1.2.3  or  package>=1.2.3  etc.
        m = re.match(r'^([A-Za-z0-9_\-\.]+)([><=!][^,]*)', dep.strip())
        if m:
            name = m.group(1).lower().replace("-", "-").replace("_", "-")
            versions[name] = dep.strip()
    return versions


def _version_matches_constraint(pinned: str, constraint: str) -> bool:
    """
    Naively check whether a pinned version string satisfies a constraint.
    Only handles >=X.Y.Z and ==X.Y.Z for simplicity; extend as needed.
    """
    try:
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
        # Extract just the version number from the pinned dep string
        m = re.search(r'[=<>!]+([\d\.]+)', pinned)
        if not m:
            return False
        ver = Version(m.group(1))
        return ver in SpecifierSet(constraint)
    except ImportError:
        # packaging not available — do a simple string check
        m = re.search(r'[=<>!]+([\d\.]+)', pinned)
        if not m:
            return False
        pinned_ver = m.group(1)
        # Crude: just check if the constraint major.minor matches
        c_m = re.search(r'([\d\.]+)', constraint)
        if not c_m:
            return True  # can't evaluate, assume applies
        c_ver = c_m.group(1)
        return pinned_ver >= c_ver


def _scan_files(scan_dirs: list[Path]) -> list[Path]:
    """Collect all .py files under the given directories."""
    files: list[Path] = []
    for d in scan_dirs:
        files.extend(d.rglob("*.py"))
    return sorted(files)


# --------------------------------------------------------------------------- #
# Core check
# --------------------------------------------------------------------------- #
def run_check(
    pyproject_path: Path,
    rules_path: Path,
    scan_dirs: list[Path],
) -> list[dict]:
    """Return a list of violation dicts."""
    pinned = _parse_pinned_versions(pyproject_path)

    with open(rules_path) as f:
        rules_data = yaml.safe_load(f)

    rules = rules_data.get("rules", [])
    py_files = _scan_files(scan_dirs)
    violations: list[dict] = []

    for rule in rules:
        pkg = rule["package"].lower().replace("_", "-")
        constraint = rule.get("version_constraint", "")
        patterns = rule.get("forbidden_patterns", [])

        # Find the pinned version for this package
        pinned_ver_str = pinned.get(pkg, "")
        if not pinned_ver_str:
            continue  # package not in project — skip

        # Check if the pinned version falls under this rule's constraint
        if constraint and not _version_matches_constraint(pinned_ver_str, constraint):
            continue  # rule doesn't apply to this version

        # Scan files for forbidden patterns
        for pattern_entry in patterns:
            pattern = pattern_entry["pattern"]
            message = pattern_entry["message"].strip()
            compiled = re.compile(pattern)

            for py_file in py_files:
                try:
                    content = py_file.read_text(encoding="utf-8")
                except OSError:
                    continue

                for lineno, line in enumerate(content.splitlines(), start=1):
                    if compiled.search(line):
                        violations.append({
                            "package": pkg,
                            "pinned": pinned_ver_str,
                            "file": str(py_file),
                            "line": lineno,
                            "code": line.strip(),
                            "message": message,
                        })

    return violations


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _build_report(violations: list[dict], pinned: dict[str, str]) -> str:
    lines = [
        "## 🔍 Version Compliance Report",
        "",
        "Generated by `agent-version-check` — checks API usage against pinned library versions.",
        "",
    ]

    if not violations:
        lines += [
            "### ✅ All checks passed",
            "",
            "No API patterns incompatible with pinned dependency versions were found.",
        ]
        return "\n".join(lines)

    lines += [
        f"### 🔴 {len(violations)} violation(s) found",
        "",
        "The following API usage patterns are incompatible with the pinned library versions",
        "in `pyproject.toml`. **Merge is blocked** until these are resolved.",
        "",
    ]

    for v in violations:
        lines += [
            f"#### `{v['package']}` ({v['pinned']})",
            "",
            f"**File:** `{v['file']}` line {v['line']}",
            "",
            f"```python",
            v["code"],
            "```",
            "",
            f"**Issue:** {v['message']}",
            "",
            "---",
            "",
        ]

    lines += [
        "### How to fix",
        "",
        "1. Update the flagged code to match the API of the pinned library version.",
        "2. If you intentionally changed the library version, update",
        "   `.github/agents/version_rules.yaml` in this same PR.",
        "3. Re-run this check via `workflow_dispatch` on `agent-version-check`.",
    ]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Version compliance checker")
    parser.add_argument("--pyproject", required=True, type=Path)
    parser.add_argument("--rules", required=True, type=Path)
    parser.add_argument("--scan-dir", dest="scan_dirs", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    violations = run_check(args.pyproject, args.rules, args.scan_dirs)
    pinned = _parse_pinned_versions(args.pyproject)
    report = _build_report(violations, pinned)

    args.output.write_text(report, encoding="utf-8")
    print(report)

    if violations:
        print(f"\n{len(violations)} violation(s) found. Exiting with code 1.", file=sys.stderr)
        return 1

    print("\nAll version compliance checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
