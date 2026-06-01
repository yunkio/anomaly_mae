"""
Exathlon dataset preprocessing — download + extract 19 FScustom features.

Downloads all 93 trace zips from the Exathlon GitHub repo, extracts the 19
FScustom features (defined in Jacob et al. VLDB 2021), generates point-level
binary labels for disturbed traces using ground_truth.csv (RCI ∪ EEI), and
deletes the original zip + full CSV after each trace is processed.

Output structure:
    ./dataset/Exathlon/
        preprocess.py            (this script)
        ground_truth.csv         (downloaded raw)
        app1/
            {trace_name}.csv     (columns: t, label, f0...f18)
        app2/
            ...
        ...

FScustom 19 features (Jacob et al. 2021, src/features/spark_alteration.py):
  Group 1 (Identity, 3): 3 driver streaming delay metrics
  Group 2 (1-Difference, 8): 4 driver counters + driver mem + driver heap + 4 CPU idle
  Group 3 (Executor-avg + 1-Difference, 6): 6 executor metrics averaged across 5 execs

Usage:
    python preprocess.py
"""
import io
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
GT_URL = "https://raw.githubusercontent.com/exathlonbenchmark/exathlon/master/data/raw/ground_truth.zip"
TREE_API = "https://api.github.com/repos/exathlonbenchmark/exathlon/git/trees/master?recursive=1"
RAW_URL_TMPL = "https://raw.githubusercontent.com/exathlonbenchmark/exathlon/master/data/raw/app{app}/{trace_name}.zip"

TYPE_CODE_TO_NAME = {
    0: "undisturbed",
    1: "bursty_input",
    2: "bursty_input_crash",
    3: "stalled_input",
    4: "cpu_contention",
    5: "process_failure",
}

# Substring patterns for the 19 FScustom features.
# Driver columns have an app-specific prefix (e.g. "driver_benchmark_userclicks_<app>_..."),
# so we match by stable suffix substring.
G1_IDENTITY = [
    "StreamingMetrics_streaming_lastCompletedBatch_processingDelay_value",
    "StreamingMetrics_streaming_lastCompletedBatch_schedulingDelay_value",
    "StreamingMetrics_streaming_lastCompletedBatch_totalDelay_value",
]
G2_DIFF_DRIVER_COUNTERS = [
    "StreamingMetrics_streaming_totalCompletedBatches_value",
    "StreamingMetrics_streaming_totalProcessedRecords_value",
    "StreamingMetrics_streaming_totalReceivedRecords_value",
    "StreamingMetrics_streaming_lastReceivedBatch_records_value",
]
G2_DIFF_OTHER = [
    "driver_BlockManager_memory_memUsed_MB_value",
    "driver_jvm_heap_used_value",
    "node5_CPU_ALL_Idle%",
    "node6_CPU_ALL_Idle%",
    "node7_CPU_ALL_Idle%",
    "node8_CPU_ALL_Idle%",
]
G3_EXEC_AVG_DIFF = [
    "executor_filesystem_hdfs_write_ops_value",
    "executor_cpuTime_count",
    "executor_runTime_count",
    "executor_shuffleRecordsRead_count",
    "executor_shuffleRecordsWritten_count",
    "jvm_heap_used_value",  # Group 3 last item per spec
]
EXECUTOR_IDS = [1, 2, 3, 4, 5]

FEATURE_NAMES = (
    [f"processingDelay", f"schedulingDelay", f"totalDelay"]
    + [f"diff_totalCompletedBatches", f"diff_totalProcessedRecords",
       f"diff_totalReceivedRecords", f"diff_lastReceivedBatch_records"]
    + [f"diff_driver_memUsed_MB", f"diff_driver_jvm_heap_used"]
    + [f"diff_node{n}_CPU_Idle" for n in (5, 6, 7, 8)]
    + [f"diff_avg_exec_hdfs_write_ops", f"diff_avg_exec_cpuTime",
       f"diff_avg_exec_runTime", f"diff_avg_exec_shuffleRecordsRead",
       f"diff_avg_exec_shuffleRecordsWritten", f"diff_avg_exec_jvm_heap_used"]
)
assert len(FEATURE_NAMES) == 19


def find_col(cols, pattern):
    """Find the unique column matching `pattern` (substring search)."""
    matches = [c for c in cols if pattern in c]
    if len(matches) == 0:
        return None
    if len(matches) > 1:
        # Prefer column ending with this pattern
        ends = [c for c in matches if c.endswith(pattern)]
        if len(ends) == 1:
            return ends[0]
        raise ValueError(f"Ambiguous column for pattern '{pattern}': {matches}")
    return matches[0]


def extract_19_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 19-feature DataFrame from raw 2,283-column CSV."""
    cols = list(df.columns)
    out = pd.DataFrame({"t": df["t"].values})

    # Group 1: 3 identity columns (driver streaming delays)
    for i, pat in enumerate(G1_IDENTITY):
        col = find_col(cols, pat)
        out[FEATURE_NAMES[i]] = df[col].values if col else np.nan

    # Group 2 (4 driver counters): 1-difference
    for j, pat in enumerate(G2_DIFF_DRIVER_COUNTERS):
        col = find_col(cols, pat)
        vals = df[col].values if col else np.full(len(df), np.nan)
        diff = np.diff(vals, prepend=vals[0])
        out[FEATURE_NAMES[3 + j]] = diff

    # Group 2 (driver memory, heap, node CPU idle): 1-difference
    for k, pat in enumerate(G2_DIFF_OTHER):
        col = find_col(cols, pat)
        vals = df[col].values if col else np.full(len(df), np.nan)
        diff = np.diff(vals, prepend=vals[0])
        out[FEATURE_NAMES[7 + k]] = diff

    # Group 3 (executor average + 1-difference) for 5 specific exec metrics
    # Then 1 for jvm_heap_used_value
    for m, metric in enumerate(G3_EXEC_AVG_DIFF[:5]):
        exec_cols = [f"{eid}_{metric}" for eid in EXECUTOR_IDS]
        exec_cols = [c for c in exec_cols if c in cols]
        if not exec_cols:
            out[FEATURE_NAMES[13 + m]] = np.nan
            continue
        avg = df[exec_cols].mean(axis=1).values
        diff = np.diff(avg, prepend=avg[0])
        out[FEATURE_NAMES[13 + m]] = diff
    # Last G3 item: jvm_heap_used_value across executors
    metric = G3_EXEC_AVG_DIFF[5]  # "jvm_heap_used_value"
    exec_cols = [f"{eid}_{metric}" for eid in EXECUTOR_IDS]
    exec_cols = [c for c in exec_cols if c in cols]
    if exec_cols:
        avg = df[exec_cols].mean(axis=1).values
        diff = np.diff(avg, prepend=avg[0])
        out[FEATURE_NAMES[18]] = diff
    else:
        out[FEATURE_NAMES[18]] = np.nan

    return out


def add_labels(out_df: pd.DataFrame, trace_name: str, gt_df: pd.DataFrame) -> pd.DataFrame:
    """Add point-level binary label column (0=normal, 1=anomaly).

    Anomaly = points in [root_cause_start, extended_effect_end] for each instance.
    """
    out_df = out_df.copy()
    labels = np.zeros(len(out_df), dtype=np.int8)
    rows = gt_df[gt_df["trace_name"] == trace_name]
    if len(rows) == 0:
        # undisturbed → all 0
        out_df["label"] = labels
        return out_df[["t", "label"] + FEATURE_NAMES]
    timestamps = out_df["t"].values
    for _, r in rows.iterrows():
        rci_s = r["root_cause_start"]
        eei_e = r["extended_effect_end"]
        if pd.isna(eei_e):
            eei_e = r["root_cause_end"]
        mask = (timestamps >= rci_s) & (timestamps <= eei_e)
        labels[mask] = 1
    out_df["label"] = labels
    return out_df[["t", "label"] + FEATURE_NAMES]


def download_ground_truth() -> pd.DataFrame:
    """Download ground_truth.zip and return DataFrame."""
    gt_csv_path = HERE / "ground_truth.csv"
    if gt_csv_path.exists():
        return pd.read_csv(gt_csv_path)
    print("[GT] downloading ground_truth.zip ...", flush=True)
    with urllib.request.urlopen(GT_URL) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as h:
            gt = pd.read_csv(h)
    gt.to_csv(gt_csv_path, index=False)
    print(f"[GT] saved {gt_csv_path} ({len(gt)} rows)", flush=True)
    return gt


def list_all_traces() -> list:
    """List all 93 trace files via GitHub git-tree API."""
    print("[LIST] fetching trace file list ...", flush=True)
    with urllib.request.urlopen(TREE_API) as r:
        tree = json.load(r)
    zips = [t for t in tree["tree"]
            if t["path"].startswith("data/raw/app") and t["path"].endswith(".zip")]
    out = []
    for z in zips:
        name = z["path"].split("/")[-1][:-4]
        app = int(name.split("_")[0])
        ttype = int(name.split("_")[1])
        out.append({"name": name, "app": app, "type": TYPE_CODE_TO_NAME[ttype]})
    print(f"[LIST] found {len(out)} traces", flush=True)
    return out


def process_trace(trace_info: dict, gt_df: pd.DataFrame, tree=None) -> bool:
    """Download one trace zip, extract 19 features, save reduced CSV, delete original.

    Handles two storage layouts in the upstream repo:
      (a) Flat:   data/raw/app{N}/{trace_name}.zip          (single-volume zip)
      (b) Nested: data/raw/app{N}/{trace_name}/             (multi-volume split zip:
                    {trace_name}.z01, .z02, ..., {trace_name}.zip)
                  → requires 7z (p7zip) for proper extraction.

    Returns True on success, False on download error.
    """
    name = trace_info["name"]
    app = trace_info["app"]
    out_dir = HERE / f"app{app}"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}.csv"

    if out_path.exists():
        return True  # skip already-processed

    # Try flat layout first
    url = RAW_URL_TMPL.format(app=app, trace_name=name)
    try:
        with urllib.request.urlopen(url) as r:
            zip_data = r.read()
        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as h:
                df = pd.read_csv(h, low_memory=False)
    except urllib.error.HTTPError as e:
        if e.code != 404 or tree is None:
            print(f"  ✗ download fail {name}: {e}", flush=True)
            return False
        # Fall back to nested split-zip layout (root-cause fix)
        df = _process_split_zip(name, app, tree)
        if df is None:
            return False
    except (zipfile.BadZipFile, ValueError) as e:
        print(f"  ✗ zip parse fail {name}: {e}", flush=True)
        return False

    # Extract 19 features + add label
    features = extract_19_features(df)
    out_df = add_labels(features, name, gt_df)
    out_df.to_csv(out_path, index=False)
    del df, features
    return True


def _process_split_zip(name: str, app: int, tree: list) -> pd.DataFrame:
    """Download all split parts and extract via 7z (which handles multi-volume zips).

    Returns the loaded raw DataFrame or None on failure.
    """
    import subprocess
    import shutil
    import tempfile

    seven_zip = shutil.which("7z") or "/home/ykio/anaconda3/envs/dc_vis/bin/7z"
    if not Path(seven_zip).exists() and not shutil.which("7z"):
        print(f"  ✗ 7z not found — install p7zip-full to handle split zips", flush=True)
        return None

    parts = sorted(
        [t for t in tree if t["path"].startswith(f"data/raw/app{app}/{name}/")],
        key=lambda x: x["path"],
    )
    if not parts:
        print(f"  ✗ no split parts found for {name}", flush=True)
        return None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for p in parts:
            fname = p["path"].split("/")[-1]
            url = f"https://raw.githubusercontent.com/exathlonbenchmark/exathlon/master/{p['path']}"
            try:
                with urllib.request.urlopen(url) as r:
                    (tmp / fname).write_bytes(r.read())
            except Exception as e:
                print(f"  ✗ split part {fname} download fail: {e}", flush=True)
                return None
        # Extract with 7z (handles multi-volume natively)
        zip_path = tmp / f"{name}.zip"
        r = subprocess.run(
            [seven_zip, "x", str(zip_path), f"-o{tmp}/extract", "-y"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  ✗ 7z extract fail: {r.stderr[:200]}", flush=True)
            return None
        csv_path = tmp / "extract" / f"{name}.csv"
        return pd.read_csv(csv_path, low_memory=False)


def main():
    print(f"Output directory: {HERE}")
    HERE.mkdir(exist_ok=True)
    gt_df = download_ground_truth()
    traces = list_all_traces()

    # Cache git tree for split-zip fallback
    print("[TREE] fetching git tree for nested-path fallback ...", flush=True)
    with urllib.request.urlopen(TREE_API) as r:
        tree = json.load(r)["tree"]

    n_ok, n_fail = 0, 0
    for i, t in enumerate(traces):
        print(f"[{i+1}/{len(traces)}] {t['name']} ({t['type']})", flush=True)
        ok = process_trace(t, gt_df, tree=tree)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone. Success: {n_ok}, Failed: {n_fail}")
    print(f"Final directory: {HERE}")


if __name__ == "__main__":
    main()
