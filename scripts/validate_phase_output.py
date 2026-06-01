#!/usr/bin/env python3
"""
Pipeline output validator. Called by report-team-lead after each phase.

Usage:
    python scripts/validate_phase_output.py phase1_json   ./temp/p1_raw_data.json
    python scripts/validate_phase_output.py phase1_md     ./temp/p1_statistician_stats.md
    python scripts/validate_phase_output.py phase2_single ./temp/p2_exp_11_analysis_raw.md
    python scripts/validate_phase_output.py phase2_cross  ./temp/p2_dl_analyst_insights.md
    python scripts/validate_phase_output.py phase3_scores ./temp/p3_critical_reviewer_critique.md
    python scripts/validate_phase_output.py phase4_file   ./temp/p4_hub_overview.md

Exit code 0 = PASS, 1 = FAIL (prints missing items to stderr)
"""

import json
import sys
import os
import re
import yaml


def validate_phase1_json(path):
    """Validate the raw JSON data from statistician."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # Check top-level structure
    if "experiments" not in data:
        errors.append("Missing top-level key: 'experiments'")
        return errors

    experiments = data["experiments"]
    if len(experiments) < 5:
        errors.append(f"Expected at least 5 experiments, found {len(experiments)}")

    required_datasets = ["simulation", "simulation_complex", "SWaT_A1A2"]

    for exp_id, exp_data in experiments.items():
        # Check config
        if "config" not in exp_data:
            errors.append(f"{exp_id}: missing 'config'")
            continue

        if "datasets" not in exp_data:
            errors.append(f"{exp_id}: missing 'datasets'")
            continue

        for ds_name in required_datasets:
            if ds_name not in exp_data["datasets"]:
                errors.append(f"{exp_id}: missing dataset '{ds_name}'")
                continue

            ds = exp_data["datasets"][ds_name]

            # Check epoch trajectories
            if "epoch_trajectories" not in ds:
                errors.append(f"{exp_id}/{ds_name}: missing 'epoch_trajectories'")
            elif len(ds["epoch_trajectories"]) < 5:
                errors.append(
                    f"{exp_id}/{ds_name}: epoch_trajectories has only "
                    f"{len(ds['epoch_trajectories'])} entries (need >= 5)"
                )
            else:
                # Check first trajectory entry has required fields
                first = ds["epoch_trajectories"][0]
                for field in ["epoch", "prc"]:
                    if field not in first:
                        errors.append(
                            f"{exp_id}/{ds_name}: trajectory missing field '{field}'"
                        )

            # Check best epoch
            if "best_epoch" not in ds:
                errors.append(f"{exp_id}/{ds_name}: missing 'best_epoch'")
            elif "epoch" not in ds.get("best_epoch", {}):
                errors.append(f"{exp_id}/{ds_name}: best_epoch missing 'epoch' field")

            # Check degradation
            if "degradation" not in ds:
                errors.append(f"{exp_id}/{ds_name}: missing 'degradation'")

            # Check components
            if "components" not in ds:
                errors.append(f"{exp_id}/{ds_name}: missing 'components'")

    # Check cross_experiment
    if "cross_experiment" not in data:
        errors.append("Missing top-level key: 'cross_experiment'")

    return errors


def validate_phase1_md(path):
    """Validate statistician markdown summary."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    with open(path) as f:
        content = f.read()

    size_kb = len(content.encode("utf-8")) / 1024
    if size_kb < 5:
        errors.append(f"File too small: {size_kb:.1f}KB (minimum 5KB)")

    return errors


def validate_phase2_single_experiment(path):
    """Validate dl-analyst SINGLE experiment analysis."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    with open(path) as f:
        content = f.read()

    size_kb = len(content.encode("utf-8")) / 1024
    if size_kb < 3:
        errors.append(f"File too small: {size_kb:.1f}KB (minimum 3KB for single-experiment analysis)")

    # Check for named phenomenon (key quality indicator)
    phenomenon_indicators = [
        "현상", "phenomenon", "패턴", "pattern",
        "사망", "death", "collapse", "spike", "transition",
        "immunity", "reversal", "역전", "지연", "delay",
    ]
    has_phenomena = any(ind.lower() in content.lower() for ind in phenomenon_indicators)
    if not has_phenomena:
        errors.append("No named phenomenon detected (expected keywords like 'phenomenon', 'death', 'collapse', 'spike')")

    # Check for temporal dynamics keywords
    temporal_indicators = [
        "epoch", "에포크", "trajectory", "궤적", "temporal", "시간",
        "dynamics", "역학", "curve", "곡선",
    ]
    has_temporal = any(ind.lower() in content.lower() for ind in temporal_indicators)
    if not has_temporal:
        errors.append("No temporal dynamics detected (expected keywords like 'epoch', 'trajectory', 'dynamics')")

    return errors


def validate_phase2_cross(path):
    """Validate dl-analyst cross-experiment analysis."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    with open(path) as f:
        content = f.read()

    size_kb = len(content.encode("utf-8")) / 1024
    if size_kb < 10:
        errors.append(f"File too small: {size_kb:.1f}KB (minimum 10KB for cross-experiment analysis)")

    # Check required analysis types
    checks = {
        "interaction": ["교호작용", "interaction", "reversal", "역전"],
        "distillation": ["증류", "distillation", "teacher", "crossover", "추월"],
        "dual_epoch": ["Best PRC", "Final PRC", "dual", "peak", "stability"],
        "cross_domain": ["과적합", "overfitting", "도메인", "domain", "degradation", "열화"],
    }

    for check_name, keywords in checks.items():
        found = any(kw.lower() in content.lower() for kw in keywords)
        if not found:
            errors.append(f"Missing analysis type '{check_name}': none of {keywords} found")

    return errors


def validate_phase3_scores(path):
    """Validate critical-reviewer scores and compute decision."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    with open(path) as f:
        content = f.read()

    # Extract YAML frontmatter
    yaml_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not yaml_match:
        return ["No YAML frontmatter found"]

    try:
        meta = yaml.safe_load(yaml_match.group(1))
    except yaml.YAMLError as e:
        return [f"Invalid YAML: {e}"]

    if "scores" not in meta:
        return ["Missing 'scores' block in YAML frontmatter"]

    scores = meta["scores"]

    # Required score items
    must_pass_items = [
        "temporal_dynamics",
        "interaction_effects",
        "causal_mechanisms",
        "distillation_analysis",
        "dual_epoch",
        "cross_domain",
        "interpretive_depth",
    ]

    all_items = must_pass_items + [
        "named_phenomena",
        "visualization_quality",
        "evidence_chains",
        "logical_consistency",
        "academic_rigor",
    ]

    # Check all items present
    for item in all_items:
        if item not in scores:
            errors.append(f"Missing score for '{item}'")
        elif not isinstance(scores[item], (int, float)):
            errors.append(f"Score for '{item}' is not a number: {scores[item]}")
        elif not (1 <= scores[item] <= 5):
            errors.append(f"Score for '{item}' out of range [1-5]: {scores[item]}")

    if errors:
        return errors

    # Compute decision
    must_pass_scores = [scores[item] for item in must_pass_items]
    all_scores = [scores[item] for item in all_items if item in scores]

    failed_must_pass = [
        (item, scores[item])
        for item in must_pass_items
        if scores[item] < 3
    ]
    mean_score = sum(all_scores) / len(all_scores)

    if failed_must_pass:
        decision = "REJECTED"
        reason = (
            f"Must-pass items below threshold (< 3): "
            + ", ".join(f"{item}={score}" for item, score in failed_must_pass)
        )
    elif mean_score < 3.5:
        decision = "REJECTED"
        reason = f"Mean score {mean_score:.2f} below threshold 3.5"
    else:
        decision = "APPROVED"
        reason = f"All must-pass items >= 3, mean score {mean_score:.2f} >= 3.5"

    # Print decision info to stdout (team-lead will read this)
    print(f"DECISION: {decision}")
    print(f"REASON: {reason}")
    print(f"MEAN_SCORE: {mean_score:.2f}")
    if failed_must_pass:
        print(f"FAILED_ITEMS: {', '.join(f'{i}={s}' for i, s in failed_must_pass)}")

    # For REJECTED, exit code 1
    if decision == "REJECTED":
        errors.append(f"AUTO_REJECTED: {reason}")

    return errors


def validate_phase4_file(path):
    """Validate a single academic-writer output file."""
    errors = []

    if not os.path.exists(path):
        return ["File does not exist"]

    with open(path) as f:
        content = f.read()

    size_kb = len(content.encode("utf-8")) / 1024

    # Different minimums based on file type
    basename = os.path.basename(path)
    if "hub" in basename:
        min_kb = 3
    elif "exp_" in basename:
        min_kb = 5
    elif "comparison" in basename:
        min_kb = 8
    elif "manifest" in basename or "draft" in basename:
        min_kb = 1
    else:
        min_kb = 3

    if size_kb < min_kb:
        errors.append(f"File too small: {size_kb:.1f}KB (minimum {min_kb}KB for {basename})")

    # Check for insight-driven visualization (no bulk image dumps)
    image_refs = re.findall(r"!\[.*?\]\(.*?\)", content)
    if len(image_refs) > 30:
        errors.append(
            f"Too many image references ({len(image_refs)}). "
            "Suspect bulk dump. Each image should be insight-linked."
        )

    return errors


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <check_type> <file_path>", file=sys.stderr)
        sys.exit(2)

    check_type = sys.argv[1]
    file_path = sys.argv[2]

    validators = {
        "phase1_json": validate_phase1_json,
        "phase1_md": validate_phase1_md,
        "phase2_single": validate_phase2_single_experiment,
        "phase2_cross": validate_phase2_cross,
        "phase3_scores": validate_phase3_scores,
        "phase4_file": validate_phase4_file,
    }

    if check_type not in validators:
        print(f"Unknown check type: {check_type}", file=sys.stderr)
        print(f"Available: {', '.join(validators.keys())}", file=sys.stderr)
        sys.exit(2)

    errors = validators[check_type](file_path)

    if errors:
        print(f"FAIL: {file_path}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"PASS: {file_path}")
        sys.exit(0)


if __name__ == "__main__":
    main()
