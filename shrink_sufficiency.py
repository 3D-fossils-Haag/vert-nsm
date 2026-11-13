#!/usr/bin/env python3
"""
shrink_sufficiency.py: turn a large sufficiency.csv into a small Parquet file and print
the summaries, using DuckDB (streams the CSV, never loads it all into memory).

    python shrink_sufficiency.py sufficiency.csv

Writes sufficiency.parquet next to it (landmark names shortened to their numbers,
e.g. "scratch_lm-1 scratch_lm-5" -> "1 5"). Open it later with
duckdb.sql("SELECT * FROM 'sufficiency.parquet' ...") or pandas.read_parquet.
"""
import os
import sys

import duckdb

GOOD = 2.0      # alignment error counted as good, % of bone size

src = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
dst = os.path.splitext(src)[0] + ".parquet"
con = duckdb.connect()

con.sql(f"""
    COPY (
        SELECT specimen, k::TINYINT AS k, jitter::REAL AS jitter,
               regexp_replace(landmarks, '[^0-9 ]+', '', 'g') AS landmarks,
               spread1::REAL AS spread1, spread2::REAL AS spread2, spread3::REAL AS spread3,
               rms_err_pct::REAL AS rms_err_pct, rot_err_deg::REAL AS rot_err_deg,
               scale_err_pct::REAL AS scale_err_pct
        FROM read_csv_auto('{src}')
    ) TO '{dst}' (FORMAT parquet, COMPRESSION zstd)
""")
print(f"{src} ({os.path.getsize(src) / 1e9:.2f} GB) -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)\n")

t = f"'{dst}'"
print(con.sql(f"SELECT count(*) AS rows, count(DISTINCT specimen) AS specimens FROM {t}"))

print(f"Alignment error (RMS, % of bone size) by number of landmarks:")
con.sql(f"""
    SELECT k, round(median(rms_err_pct), 2) AS median, round(quantile_cont(rms_err_pct, 0.9), 2) AS p90,
           round(avg((rms_err_pct < {GOOD})::INT), 2) AS share_under_{int(GOOD)}pct, count(*) AS n
    FROM {t} WHERE jitter > 0 GROUP BY k ORDER BY k
""").show(max_rows=40)

print(con.sql(f"""
    SELECT round(median(rms_err_pct), 2) AS floor_median_pct
    FROM {t} WHERE jitter = 0
"""))

print("Triplets by spread (2nd axis, quartiles):")
print(con.sql(f"""
    SELECT q AS spread_quartile, round(min(spread2), 3) AS spread2_from, round(max(spread2), 3) AS spread2_to,
           round(median(rms_err_pct), 2) AS median_err_pct
    FROM (SELECT *, ntile(4) OVER (ORDER BY spread2) AS q FROM {t} WHERE k = 3 AND jitter > 0)
    GROUP BY q ORDER BY q
"""))

print("Best triplets across specimens (landmark numbers):")
print(con.sql(f"""
    SELECT landmarks, round(median(rms_err_pct), 2) AS median_err_pct,
           round(quantile_cont(rms_err_pct, 0.9), 2) AS p90, count(*) AS specimens
    FROM {t} WHERE k = 3 AND jitter > 0
    GROUP BY landmarks ORDER BY median_err_pct LIMIT 10
"""))