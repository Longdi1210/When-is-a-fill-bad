from __future__ import annotations

import json
import time
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from fillbad.real_btc import PARQUET_DIR, RAW_CSV, parse_timestamps


def convert_csv_to_partitioned_parquet(chunk_size: int = 100_000, force: bool = False) -> dict:
    start = time.perf_counter()
    if PARQUET_DIR.exists() and (PARQUET_DIR / "manifest.csv").exists() and not force:
        report_path = PARQUET_DIR / "conversion_report.json"
        if report_path.exists():
            return json.loads(report_path.read_text())
    if PARQUET_DIR.exists() and force:
        shutil.rmtree(PARQUET_DIR)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    sample_before = None
    sample_after = None

    for chunk_id, chunk in enumerate(pd.read_csv(RAW_CSV, chunksize=chunk_size)):
        timestamps = parse_timestamps(chunk["system_time"])
        chunk = chunk.assign(system_time=timestamps.astype("string"), date=timestamps.dt.strftime("%Y-%m-%d"))
        if sample_before is None:
            sample_before = chunk.drop(columns=["date"]).head(3).to_dict(orient="records")
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=PARQUET_DIR,
            partition_cols=["date"],
            compression="zstd",
            basename_template=f"part-{chunk_id:05d}-{{i}}.parquet",
        )
        rows_written += len(chunk)

    dataset = ds.dataset(PARQUET_DIR, format="parquet", partitioning="hive", exclude_invalid_files=True)
    row_count = dataset.count_rows()
    preview = dataset.head(3, columns=["system_time", "midpoint", "spread"]).to_pandas()
    sample_after = preview.to_dict(orient="records")
    fragments = list(dataset.get_fragments())
    manifest_rows = []
    total_parquet_size = 0
    for fragment in fragments:
        path = Path(fragment.path)
        size = path.stat().st_size
        total_parquet_size += size
        table = pq.read_table(path, columns=["system_time"])
        ts = parse_timestamps(table["system_time"].to_pandas())
        manifest_rows.append(
            {
                "file": str(path),
                "row_count": int(table.num_rows),
                "start_time_utc": ts.min().isoformat(),
                "end_time_utc": ts.max().isoformat(),
                "size_bytes": int(size),
            }
        )
    manifest = pd.DataFrame(manifest_rows).sort_values(["start_time_utc", "file"])
    manifest.to_csv(PARQUET_DIR / "manifest.csv", index=False)
    report = {
        "raw_csv": str(RAW_CSV),
        "parquet_dir": str(PARQUET_DIR),
        "rows_written": int(rows_written),
        "rows_read_back": int(row_count),
        "row_count_verified": bool(rows_written == row_count),
        "partition_count": int(manifest["file"].nunique()),
        "parquet_size_bytes": int(total_parquet_size),
        "parquet_size_gb": total_parquet_size / 1e9,
        "conversion_runtime_seconds": time.perf_counter() - start,
        "sample_before": sample_before,
        "sample_after": sample_after,
    }
    (PARQUET_DIR / "conversion_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    print(json.dumps(convert_csv_to_partitioned_parquet(), indent=2, default=str))


if __name__ == "__main__":
    main()
