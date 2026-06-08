"""
arch_review_agent.py — Multi-persona architecture review agent.

Reviews PRs from five simultaneous lenses and optionally auto-updates
ARCHITECTURE.md after a merge to main.

Modes:
  review  (default): Generate a PR review comment + a patch suggestion file.
  apply:             Apply the patch file to ARCHITECTURE.md (post-merge mode).

Usage — PR review mode:
    python arch_review_agent.py \\
        --diff /tmp/arch_diff.txt \\
        --source /tmp/full_source.txt \\
        --current-arch ARCHITECTURE.md \\
        --output-review /tmp/arch_review.md \\
        --output-patch /tmp/arch_patch.md

Usage — post-merge apply mode:
    python arch_review_agent.py \\
        --apply-patch /tmp/arch_patch.md \\
        --arch-file ARCHITECTURE.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm_client  # noqa: E402

# --------------------------------------------------------------------------- #
# Personas
# --------------------------------------------------------------------------- #
PERSONAS = {
    "App Developer": (
        "Focus on: API contract changes (FastAPI endpoints, Pydantic schemas), "
        "Streamlit dashboard dependencies, input schema drift (feature list changes), "
        "backward compatibility of /predict and /predict_batch endpoints."
    ),
    "Infrastructure": (
        "Focus on: Docker image layer order and cache efficiency, bind-mount strategy "
        "($(pwd):/app), volume changes, resource constraints, Dockerfile base image, "
        "uv environment location (/opt/venv vs /app)."
    ),
    "DevOps": (
        "Focus on: GitHub Actions workflow triggers and permissions scope, secret "
        "handling (never print, use masks), caching strategy (GHA cache, uv cache), "
        "artifact upload retention, required status checks, self-hosted runner routing."
    ),
    "MLOps": (
        "Focus on: DVC stage changes (new deps/outs, removed stages), model artifact "
        "paths (models/best_model.pkl), feature store versioning (v1/v2), monitoring "
        "logic (PSI thresholds, Evidently presets), config.py as single source of truth, "
        "MLflow experiment tracking, CodeCarbon integration."
    ),
    "Cloud": (
        "Focus on: GHCR image naming and tag strategy, Evidently Cloud token/project "
        "handling, any introduced cloud provider calls (GCP/AWS/Azure), egress cost "
        "implications, secrets management for cloud credentials."
    ),
}

REVIEW_SYSTEM = """
You are a {persona} reviewing changes to an MLOps project.
Your review scope: {scope}

Project context is provided. Analyse the diff and return ONLY a markdown
bullet list (no headings, no preamble) with findings from your perspective.
Each bullet must start with one of: ✅ (good), ⚠️ (warning), 🔴 (critical).
Maximum 5 bullets. Be terse and specific — reference file names and line numbers
where possible.
""".strip()

PATCH_SYSTEM = """
You are an MLOps technical writer. You have been given:
  1. The current ARCHITECTURE.md content.
  2. A set of multi-persona review findings from a PR review.
  3. The actual source changes (diff).

Produce a JSON object describing ARCHITECTURE.md sections that need updating.
Format:
{
  "updates": [
    {
      "section": "## 7. Monitoring ...",   // exact heading that needs updating
      "reason": "one-line explanation",
      "new_content": "full replacement markdown for that section"
    }
  ]
}

Rules:
  - Only include sections that genuinely changed.
  - If nothing needs updating, return {"updates": []}.
  - Preserve all existing content except what actually changed.
  - Keep the technical depth and table formats of the existing document.
  - Output ONLY the JSON object, no markdown fences, no explanation.
""".strip()


# --------------------------------------------------------------------------- #
# Review mode
# --------------------------------------------------------------------------- #
def run_review(
    diff: str,
    source: str,
    current_arch: str,
    prefer_local: bool,
) -> tuple[str, str]:
    """Return (review_comment_md, patch_json_str)."""
    diff_snip = diff[:6000]
    src_snip = source[:4000]
    arch_snip = current_arch[:5000]

    base_context = (
        f"=== SOURCE SNAPSHOT ===\n{src_snip}\n\n"
        f"=== PR DIFF ===\n{diff_snip}\n\n"
        f"=== CURRENT ARCHITECTURE.md ===\n{arch_snip}\n"
    )

    # Run all five personas
    persona_results: dict[str, str] = {}
    for persona, scope in PERSONAS.items():
        system = REVIEW_SYSTEM.format(persona=persona, scope=scope)
        print(f"[arch_review_agent] Running {persona} persona...", file=sys.stderr)
        result = llm_client.call(base_context, system=system, prefer_local=prefer_local)
        persona_results[persona] = result.strip()

    # Build the PR review comment
    comment_lines = [
        "## 🏗️ Architecture Review — 5-Lens Analysis",
        "",
        "> Generated by `agent-arch-review` — reviews every PR from five engineering perspectives.",
        "",
    ]
    for persona, findings in persona_results.items():
        comment_lines += [f"### {persona}", "", findings, ""]

    # Ask the LLM to generate the ARCHITECTURE.md patch
    patch_context = (
        f"=== CURRENT ARCHITECTURE.md ===\n{current_arch}\n\n"
        f"=== REVIEW FINDINGS ===\n"
        + "\n\n".join(f"**{p}:**\n{f}" for p, f in persona_results.items())
        + f"\n\n=== PR DIFF ===\n{diff_snip}\n"
    )
    print("[arch_review_agent] Generating ARCHITECTURE.md patch...", file=sys.stderr)
    patch_json = llm_client.call(patch_context, system=PATCH_SYSTEM, prefer_local=prefer_local)

    comment_lines += [
        "---",
        "*To re-run: Actions → Agent — Architecture Review → Run workflow.*",
    ]
    return "\n".join(comment_lines), patch_json


# --------------------------------------------------------------------------- #
# Apply mode
# --------------------------------------------------------------------------- #
def apply_patch(patch_path: Path, arch_path: Path) -> None:
    """
    Parse the JSON patch and apply section replacements to ARCHITECTURE.md.
    Silently skips sections that can't be found (safe for partial matches).
    """
    import json

    patch_text = patch_path.read_text(encoding="utf-8")
    try:
        # Strip markdown fences if the LLM wrapped the JSON
        patch_text = patch_text.strip()
        if patch_text.startswith("```"):
            patch_text = "\n".join(patch_text.split("\n")[1:])
            patch_text = patch_text.rstrip("`").strip()
        data = json.loads(patch_text)
    except json.JSONDecodeError as e:
        print(f"[arch_review_agent] Patch JSON parse error: {e} — skipping apply.", file=sys.stderr)
        return

    updates = data.get("updates", [])
    if not updates:
        print("[arch_review_agent] No ARCHITECTURE.md updates needed.", file=sys.stderr)
        return

    content = arch_path.read_text(encoding="utf-8")
    applied = 0
    for update in updates:
        heading = update.get("section", "").strip()
        new_content = update.get("new_content", "").strip()
        if not heading or not new_content:
            continue

        # Find the heading and replace until the next same-level heading
        heading_level = len(heading.split()[0])  # count leading '#'
        import re

        pattern = re.compile(
            rf"(^{re.escape(heading)}.*?)(?=^{'#' * heading_level}\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        new_content_block = new_content + "\n\n"
        content, n = pattern.subn(new_content_block, content, count=1)
        if n:
            print(f"[arch_review_agent] Updated section: {heading[:60]}", file=sys.stderr)
            applied += 1
        else:
            print(
                f"[arch_review_agent] Section not found, skipped: {heading[:60]}", file=sys.stderr
            )

    if applied:
        arch_path.write_text(content, encoding="utf-8")
        print(f"[arch_review_agent] {applied} section(s) updated in {arch_path}.", file=sys.stderr)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-persona architecture review agent")
    # Review mode args
    parser.add_argument("--diff", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--current-arch", type=Path)
    parser.add_argument("--output-review", type=Path)
    parser.add_argument("--output-patch", type=Path)
    # Apply mode args
    parser.add_argument("--apply-patch", type=Path)
    parser.add_argument("--arch-file", type=Path)
    # LLM preference
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Prefer local Ollama (self-hosted runner mode)",
    )
    args = parser.parse_args()

    if args.apply_patch and args.arch_file:
        # Post-merge apply mode
        apply_patch(args.apply_patch, args.arch_file)
        return 0

    # PR review mode
    if not all([args.diff, args.source, args.current_arch, args.output_review, args.output_patch]):
        parser.error(
            "Review mode requires: --diff --source --current-arch --output-review --output-patch"
        )

    diff = args.diff.read_text(encoding="utf-8", errors="replace")
    source = args.source.read_text(encoding="utf-8", errors="replace")
    current_arch = args.current_arch.read_text(encoding="utf-8", errors="replace")

    if not diff.strip():
        msg = "## 🏗️ Architecture Review\n\nNo relevant file changes detected.\n"
        args.output_review.write_text(msg, encoding="utf-8")
        args.output_patch.write_text('{"updates": []}', encoding="utf-8")
        return 0

    review, patch = run_review(diff, source, current_arch, args.local)
    args.output_review.write_text(review, encoding="utf-8")
    args.output_patch.write_text(patch, encoding="utf-8")
    print(review)
    return 0


if __name__ == "__main__":
    sys.exit(main())
