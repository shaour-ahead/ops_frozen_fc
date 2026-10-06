#!/usr/bin/env python3
"""
freeze_retail_forecast.py — freeze the B2B / retail forecast month by month.

Standalone twin of freeze_forecast.py, for the retail sheet only:

    Copy of Retail Forecast Model 12mr - 2026 - September 9, 2_15 PM - Productlevel FC.csv

The month in the file name ("September") is the snapshot month. From that snapshot
the next N months (default 2) are frozen -> Oct 2026 and Nov 2026, with
recorded_fetch_date = 2026-09-01. Months that are already in the frozen file are
never rewritten, so next month only the new month is appended.

Usage
-----
    # first run
    python freeze_retail_forecast.py -i "Retail Forecast Model ... September 9 ... .csv" \
        -o frozen_retail_forecast.csv

    # following months - append to the existing frozen file
    python freeze_retail_forecast.py -i "Retail Forecast Model ... October 7 ... .csv" \
        -e frozen_retail_forecast.csv

Options
-------
    -i / --input        retail forecast export, product level (.csv / .xlsx)
    -e / --existing     already frozen output file to append to
    -o / --output       where to write (default: --existing, else ./frozen_retail_forecast.csv)
    -m / --file-month   override the snapshot month, e.g. 2026-09
    -n / --horizon      how many months to freeze per snapshot (default 2)
    --dupes             sum (default) or first, when a SKU is listed twice in the sheet
    --label             value written to the label column (default: b2b fc)
    --force             re-freeze months that already exist in the frozen file
    --dry-run           show what would happen, write nothing

Output columns
--------------
    status, sku, artikelbezeichnung, category, month, label, forecasted,
    forecasted_date, recorded_fetch_date, source_file

Same schema as freeze_forecast.py, so the two frozen files can simply be stacked
(or loaded into the same BigQuery table).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #

ID_COLS = ["status", "sku", "artikelbezeichnung", "category"]

MONTH_ABBR = ["jan", "feb", "mar", "apr", "may", "jun",
              "jul", "aug", "sep", "oct", "nov", "dec"]

MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
    # German, in case the export is renamed
    "januar": 1, "februar": 2, "maerz": 3, "märz": 3, "mai": 5,
    "juni": 6, "juli": 7, "oktober": 10, "dezember": 12,
}

# 'Jan. 26' -> 'jan_26'
MONTH_COL_RE = re.compile(r"^([a-z]{3})_(\d{2})$")

DEFAULT_LABEL = "b2b fc"

NA_STRINGS = {"", "n/a", "na", "nan", "none", "null", "-", "–",
              "#ref!", "#value!", "#div/0!", "#n/a", "#name?", "#num!"}

OUTPUT_COLS = ["status", "sku", "artikelbezeichnung", "category", "month", "label",
               "forecasted", "forecasted_date", "recorded_fetch_date", "source_file"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def month_from_filename(path: Path, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Pull the snapshot month out of a file name like
    'Copy of Retail Forecast Model 12mr - 2026 - September 9, 2_15 PM - Productlevel FC.csv'.

    A 4 digit year in the name is used if present, otherwise the most recent
    occurrence of that month that is not in the future. Returns the first of the month.

    Careful: this file name also carries a model year ('- 2026 -') before the month.
    Only a year that appears *after* the month name is treated as the snapshot year,
    which for the example above gives September 2026 either way.
    """
    today = today or pd.Timestamp.today().normalize()
    stem = path.stem.replace("_", " ").replace("-", " ")

    match = None
    for m in re.finditer(r"[A-Za-zÄÖÜäöü]+", stem):
        name = m.group(0).lower().rstrip(".")
        if name in MONTH_NAMES:
            match = (name, m.end())
            break
    if match is None:
        raise ValueError(f"No month found in file name '{path.name}'. "
                         f"Pass it explicitly, e.g. --file-month 2026-09")

    month = MONTH_NAMES[match[0]]
    tail = stem[match[1]:]
    year_match = re.search(r"\b(20\d{2})\b", tail)
    if year_match:
        year = int(year_match.group(1))
    else:
        # no year in the name: take the occurrence of that month closest to today,
        # ties going to the past (an export is normally processed in its own month)
        candidates = [pd.Timestamp(year=y, month=month, day=1)
                      for y in (today.year - 1, today.year, today.year + 1)]
        ref = today.replace(day=1)
        year = min(candidates, key=lambda c: (abs((c - ref).days), c > ref)).year

    return pd.Timestamp(year=year, month=month, day=1)


def to_number(value) -> float | None:
    """German / English safe number parsing: '7.228' -> 7228, '1.234,5' -> 1234.5."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().replace("\u00a0", "").replace(" ", "")
    if s.lower() in NA_STRINGS:
        return None
    s = s.replace("€", "").replace("%", "")

    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")                       # decimal comma
    elif "." in s:
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")                    # thousands separator
    try:
        out = float(s)
    except ValueError:
        return None
    return -out if negative else out


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, dtype=str)
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


# --------------------------------------------------------------------------- #
# core transformation
# --------------------------------------------------------------------------- #

def build_long_frame(path: Path,
                     snapshot_date: pd.Timestamp,
                     dupes: str = "sum",
                     label: str = DEFAULT_LABEL) -> pd.DataFrame:
    """Read one retail export and return it in long format (all months in the file)."""
    df = read_any(path)

    # 'Jan. 26' -> 'jan_26'
    df.columns = (
        pd.Index([str(c) for c in df.columns])
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )
    df = df.loc[:, ~df.columns.duplicated()]

    if "sku" not in df.columns:
        raise ValueError(f"No 'SKU' column found in {path.name}")

    month_cols = [c for c in df.columns if MONTH_COL_RE.match(c)]
    if not month_cols:
        raise ValueError(f"No month columns (e.g. 'Jan. 26') found in {path.name}")
    month_cols.sort(key=lambda c: (MONTH_COL_RE.match(c).group(2),
                                   MONTH_ABBR.index(MONTH_COL_RE.match(c).group(1))))

    # --- ids: blank out NaN and Excel errors, drop rows without a SKU -------
    for col in ID_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("").astype(str).str.strip()
        df.loc[df[col].str.lower().isin(NA_STRINGS), col] = ""
    df = df[df["sku"].ne("")].copy()

    # --- numbers ------------------------------------------------------------
    for col in month_cols:
        df[col] = (
            pd.Series(df[col].map(to_number).values, index=df.index)
            .apply(lambda v: np.floor(v) if pd.notna(v) else np.nan)
            .astype("Float64").fillna(0).astype("Int64")
        )

    # --- a SKU listed twice -------------------------------------------------
    duplicated = df.loc[df["sku"].duplicated(keep=False), "sku"].unique()
    if len(duplicated):
        how = "summed" if dupes == "sum" else "first row kept"
        shown = ", ".join(sorted(duplicated)[:10])
        print(f"  note {len(duplicated)} SKU(s) listed more than once ({how}): {shown}"
              f"{' ...' if len(duplicated) > 10 else ''}")
        if dupes == "sum":
            agg = {c: "sum" for c in month_cols}
            agg.update({c: "first" for c in ID_COLS if c != "sku"})
            df = df.groupby("sku", as_index=False).agg(agg)
        else:
            df = df.drop_duplicates(subset="sku", keep="first")

    # --- long format --------------------------------------------------------
    long = df[ID_COLS + month_cols].melt(
        id_vars=ID_COLS, var_name="period", value_name="forecasted")
    long["forecasted_date"] = pd.to_datetime(
        long["period"].str.replace("_", "-", regex=False), format="%b-%y", errors="coerce")
    long = long[long["forecasted_date"].notna()].copy()

    long["label"] = label
    long["month"] = label + " " + long["period"].str.replace("_", "", regex=False)
    long["recorded_fetch_date"] = snapshot_date
    long["source_file"] = path.name
    long["forecasted"] = long["forecasted"].astype("Int64")
    return long[OUTPUT_COLS]


def load_existing(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame(columns=OUTPUT_COLS)
    df = read_any(Path(path))
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    df["forecasted_date"] = pd.to_datetime(df["forecasted_date"], errors="coerce")
    df["recorded_fetch_date"] = pd.to_datetime(df["recorded_fetch_date"], errors="coerce")
    df["forecasted"] = pd.to_numeric(df["forecasted"], errors="coerce").astype("Int64")
    return df[OUTPUT_COLS]


def freeze(input_path: Path,
           existing_path: Path | None = None,
           output_path: Path | None = None,
           horizon: int = 2,
           file_month: pd.Timestamp | None = None,
           force: bool = False,
           dry_run: bool = False,
           dupes: str = "sum",
           label: str = DEFAULT_LABEL) -> pd.DataFrame:

    snapshot = file_month or month_from_filename(input_path)
    targets = [(snapshot + pd.DateOffset(months=i)).normalize() for i in range(1, horizon + 1)]

    print(f"file            : {input_path.name}")
    print(f"snapshot month  : {snapshot:%Y-%m} (recorded_fetch_date = {snapshot:%Y-%m-%d})")
    print(f"freeze horizon  : {', '.join(t.strftime('%b %Y') for t in targets)}")

    existing = load_existing(existing_path)
    frozen_months = set(existing["forecasted_date"].dropna().unique()) if not existing.empty else set()
    if frozen_months:
        have = sorted(pd.to_datetime(list(frozen_months)))
        print(f"already frozen  : {', '.join(d.strftime('%b %Y') for d in have)}")

    new_months = [t for t in targets if force or t.to_datetime64() not in frozen_months]
    for t in [t for t in targets if t not in new_months]:
        print(f"  skip {t:%b %Y} - already frozen (use --force to overwrite)")
    if not new_months:
        print("nothing to add.")
        return existing

    long = build_long_frame(input_path, snapshot, dupes, label)
    available = set(long["forecasted_date"].unique())
    for t in [t for t in new_months if t.to_datetime64() not in available]:
        print(f"  warn {t:%b %Y} - no column for this month in the file")
    new_months = [t for t in new_months if t.to_datetime64() in available]
    if not new_months:
        print("nothing to add.")
        return existing

    addition = long[long["forecasted_date"].isin(new_months)].copy()

    if force and not existing.empty:
        existing = existing[~existing["forecasted_date"].isin(new_months)]

    result = pd.concat([existing, addition], ignore_index=True)
    for col in ("forecasted_date", "recorded_fetch_date"):
        result[col] = pd.to_datetime(result[col], errors="coerce")
    result = result.drop_duplicates(subset=["sku", "label", "forecasted_date"], keep="last")
    result = result.sort_values(["forecasted_date", "sku", "label"]).reset_index(drop=True)

    for t in new_months:
        part = addition[addition["forecasted_date"] == t]
        print(f"  add  {t:%b %Y} - {len(part):>5} rows, {part['sku'].nunique()} SKUs, "
              f"{int(part['forecasted'].sum()):,} units")

    out = result.copy()
    out["forecasted_date"] = out["forecasted_date"].dt.strftime("%Y-%m-%d")
    out["recorded_fetch_date"] = out["recorded_fetch_date"].dt.strftime("%Y-%m-%d")

    target_file = output_path or existing_path or Path("frozen_retail_forecast.csv")
    if dry_run:
        print(f"dry run - would write {len(out)} rows to {target_file}")
    else:
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target_file, index=False)
        print(f"written         : {target_file} ({len(out)} rows total)")
    return result


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Freeze the next N months of the retail / B2B forecast export.")
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="retail forecast export, product level (.csv/.xlsx)")
    p.add_argument("-e", "--existing", type=Path, default=None, help="already frozen output file")
    p.add_argument("-o", "--output", type=Path, default=None, help="where to write the frozen file")
    p.add_argument("-m", "--file-month", default=None, help="override snapshot month, e.g. 2026-09")
    p.add_argument("-n", "--horizon", type=int, default=2, help="months to freeze per snapshot (default 2)")
    p.add_argument("--dupes", choices=["sum", "first"], default="sum",
                   help="how to handle a SKU listed twice (default: sum)")
    p.add_argument("--label", default=DEFAULT_LABEL, help=f"label column value (default: {DEFAULT_LABEL})")
    p.add_argument("--force", action="store_true", help="overwrite months that are already frozen")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, write nothing")
    a = p.parse_args(argv)

    if not a.input.exists():
        print(f"input not found: {a.input}", file=sys.stderr)
        return 1

    file_month = None
    if a.file_month:
        file_month = pd.Timestamp(a.file_month).replace(day=1).normalize()

    freeze(a.input, a.existing, a.output, a.horizon, file_month,
           a.force, a.dry_run, a.dupes, a.label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
