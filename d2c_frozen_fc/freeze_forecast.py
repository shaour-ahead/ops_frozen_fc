#!/usr/bin/env python3
"""
freeze_forecast.py — build a historical (frozen) snapshot of the ahead product forecast.

Idea
----
Every month a new export of the forecast sheet is produced, e.g.

    Copy of Ahead Product FC 25 & 26 - June 9, 5_21 PM - FC by Channel 12mr.csv

The month in the file name ("June") is the snapshot month. From that snapshot we
freeze the forecast for the next N months (default 2) -> July 2026 and August 2026.
recorded_fetch_date is set to the first day of the snapshot month (2026-06-01).

Next month the July file is given. August is already frozen from the June file, so
only September is appended. Months that already exist in the frozen file are never
rewritten (unless --force is passed).

What it covers
--------------
This script handles the WS and AMZ channels only:

    ws total  = "WS FC <month>" + "WS INC <month>"   (the sum applies to WS only)
    amz fc    = "AMZ FC <month>"                     (taken as is, no inc counterpart)

B2B is deliberately left out, because the b2b columns in this sheet are empty.
The b2b numbers come from the retail sheet via freeze_retail_forecast.py, which
writes the same columns, so the two frozen files can simply be stacked.

Usage
-----
    # first run - creates the frozen file
    python freeze_forecast.py -i "Copy of Ahead Product FC ... June 9 ....csv" \
        -o frozen_forecast.csv

    # following months - append to the existing frozen file
    python freeze_forecast.py -i "...July 8....csv" -e frozen_forecast.csv

    # status / name / category are #REF! in this export; fill them from another sheet
    python freeze_forecast.py -i "...July 8....csv" -e frozen_forecast.csv \
        --meta "Retail Forecast Model ... July 8 ....csv"

Options
-------
    -i / --input          channel forecast export, FC by Channel (.csv / .xlsx)
    -e / --existing       already frozen output file to append to
    -o / --output         where to write (default: the --existing path, else ./frozen_forecast.csv)
    -m / --file-month     override the snapshot month, e.g. 2026-06 (if the file name is unusual)
    -n / --horizon        how many months to freeze per snapshot (default 2)
    --meta                sheet to fill empty status / name / category from, by SKU
    --force               re-freeze months that already exist in the frozen file
    --dry-run             show what would happen, write nothing

Output columns
--------------
    status, sku, artikelbezeichnung, category, month, label, forecasted,
    forecasted_date, recorded_fetch_date, source_file
    label is one of: ws total, amz fc
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

# channel prefixes taken from the sheet (lower case, after the "/" -> "-" clean up).
# b2b is deliberately NOT read here - the channel sheet has it empty, the numbers
# come from the retail sheet via freeze_retail_forecast.py.
PREFIXES = ["ws fc", "ws inc", "amz fc"]

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

# "ws fc jun26" / "amz fc dec27" ...
FC_COL_RE = re.compile(r"^(ws fc|ws inc|amz fc) ([a-z]{3})(\d{2})$")

NA_STRINGS = {"", "n/a", "na", "nan", "none", "null", "-", "–",
              "#ref!", "#value!", "#div/0!", "#n/a", "#name?"}

OUTPUT_COLS = ["status", "sku", "artikelbezeichnung", "category", "month", "label",
               "forecasted", "forecasted_date", "recorded_fetch_date", "source_file"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def month_from_filename(path: Path, today: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Pull the snapshot month out of a file name like
    'Copy of Ahead Product FC 25 & 26 - June 9, 5_21 PM - FC by Channel 12mr.csv'.

    The year is used if the name contains a 4 digit year (20xx). Otherwise the most
    recent occurrence of that month that is not in the future is taken.
    Returns the first day of that month.
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
        raise ValueError(
            f"No month found in file name '{path.name}'. Pass it explicitly, "
            f"e.g. --file-month 2026-06"
        )

    month = MONTH_NAMES[match[0]]

    # explicit 4 digit year anywhere in the name (prefer one right after the month)
    tail = stem[match[1]:]
    year_match = re.search(r"\b(20\d{2})\b", tail) or re.search(r"\b(20\d{2})\b", stem)
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
    """Robust German/English number parsing: '1.234,5' -> 1234.5, '1234' -> 1234."""
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
        # whichever comes last is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")            # German decimal comma
    elif "." in s:
        if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")          # German thousands separator
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

def build_long_frame(path: Path, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Read one forecast export and return it in long format (all months)."""
    df = read_any(path)

    # --- drop rows without a SKU -------------------------------------------
    sku_col = next((c for c in df.columns if str(c).strip().lower() == "sku"), None)
    if sku_col is None:
        raise ValueError(f"No 'SKU' column found in {path.name}")
    sku = df[sku_col].astype(str).str.strip()
    df = df[df[sku_col].notna() & (sku != "") & (~sku.str.lower().isin(NA_STRINGS))].copy()

    # --- normalise column names --------------------------------------------
    df.columns = [str(c).replace("/", "-").strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # --- keep ids + the forecast columns (all years present in the file) ----
    fc_cols = [c for c in df.columns if FC_COL_RE.match(c)]
    # stable order: year, month, prefix - same as the original nested loop
    fc_cols.sort(key=lambda c: (
        FC_COL_RE.match(c).group(3),
        MONTH_ABBR.index(FC_COL_RE.match(c).group(2)),
        PREFIXES.index(FC_COL_RE.match(c).group(1)),
    ))
    missing_ids = [c for c in ID_COLS if c not in df.columns]
    for c in missing_ids:
        df[c] = pd.NA
    df = df[ID_COLS + fc_cols].copy()

    # --- numbers ------------------------------------------------------------
    for col in fc_cols:
        df[col] = (
            df[col].map(to_number)
            .pipe(pd.Series, index=df.index)
            .apply(lambda v: np.floor(v) if pd.notna(v) else np.nan)
            .astype("Float64")
            .fillna(0)
            .astype("Int64")
        )

    for col in ID_COLS:
        df[col] = df[col].astype("string").fillna("").astype(str).str.strip()
        df.loc[df[col].str.lower().isin(NA_STRINGS), col] = ""

    # --- ws total = ws fc + ws inc --------------------------------------
    # This sum applies to WS only. AMZ has no "inc" counterpart and is carried
    # over unchanged as "amz fc".
    periods = sorted({FC_COL_RE.match(c).group(2) + FC_COL_RE.match(c).group(3)
                      for c in fc_cols})
    no_inc = []
    for period in periods:
        ws_fc, ws_inc = f"ws fc {period}", f"ws inc {period}"
        if ws_fc not in df.columns:
            continue
        if ws_inc in df.columns:
            df[f"ws total {period}"] = df[ws_fc] + df[ws_inc]
        else:
            df[f"ws total {period}"] = df[ws_fc]
            no_inc.append(period)
    if no_inc:
        print(f"  warn no 'ws inc' column for {', '.join(no_inc)} - "
              f"ws total falls back to ws fc alone")

    df = df.drop(columns=[c for c in df.columns
                          if c.startswith("ws fc ") or c.startswith("ws inc ")])

    # --- long format --------------------------------------------------------
    long = df.melt(id_vars=ID_COLS, var_name="month", value_name="forecasted")
    long[["label", "period"]] = long["month"].str.rsplit(" ", n=1, expand=True)
    long["forecasted_date"] = pd.to_datetime(long["period"], format="%b%y", errors="coerce")
    long = long.drop(columns=["period"])
    long = long[long["forecasted_date"].notna()].copy()

    long["recorded_fetch_date"] = snapshot_date
    long["source_file"] = path.name
    long["forecasted"] = long["forecasted"].astype("Int64")
    return long[OUTPUT_COLS]


def load_metadata(path: Path) -> pd.DataFrame:
    """
    Read status / artikelbezeichnung / category per SKU from a reference sheet
    (the retail product level export works well). Used only to fill the blanks
    the channel sheet leaves behind - no forecast numbers are taken from it.
    """
    df = read_any(path)
    df.columns = (
        pd.Index([str(c) for c in df.columns])
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )
    df = df.loc[:, ~df.columns.duplicated()]
    if "sku" not in df.columns:
        raise ValueError(f"No 'SKU' column found in {path.name}")

    for col in ID_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("").astype(str).str.strip()
        df.loc[df[col].str.lower().isin(NA_STRINGS), col] = ""

    return (df.loc[df["sku"].ne(""), ID_COLS]
            .replace("", pd.NA)
            .drop_duplicates(subset="sku")
            .set_index("sku"))


def apply_metadata(long: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Fill empty status / name / category from the reference sheet, by SKU."""
    long = long.copy()
    for col in ["status", "artikelbezeichnung", "category"]:
        blank = long[col].fillna("").astype(str).str.strip().eq("")
        if not blank.any():
            continue
        long.loc[blank, col] = long.loc[blank, "sku"].map(meta[col]).fillna("").values
    return long


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


def freeze(input_path: Path | None = None,
           existing_path: Path | None = None,
           output_path: Path | None = None,
           horizon: int = 2,
           file_month: pd.Timestamp | None = None,
           force: bool = False,
           dry_run: bool = False,
           meta_path: Path | None = None) -> pd.DataFrame:

    if input_path is None:
        raise ValueError("--input (the FC by Channel export) is required.")

    snapshot = file_month or month_from_filename(input_path)
    targets = [(snapshot + pd.DateOffset(months=i)).normalize() for i in range(1, horizon + 1)]

    print(f"file            : {input_path.name}")
    if meta_path:
        print(f"metadata from   : {meta_path.name}")
    print(f"snapshot month  : {snapshot:%Y-%m} (recorded_fetch_date = {snapshot:%Y-%m-%d})")
    print(f"freeze horizon  : {', '.join(t.strftime('%b %Y') for t in targets)}")

    existing = load_existing(existing_path)
    frozen_months = set(existing["forecasted_date"].dropna().unique()) if not existing.empty else set()
    if frozen_months:
        have = sorted(pd.to_datetime(list(frozen_months)))
        print(f"already frozen  : {', '.join(d.strftime('%b %Y') for d in have)}")

    new_months = [t for t in targets if force or t.to_datetime64() not in frozen_months]
    skipped = [t for t in targets if t not in new_months]
    for t in skipped:
        print(f"  skip {t:%b %Y} - already frozen (use --force to overwrite)")
    if not new_months:
        print("nothing to add.")
        return existing

    long = build_long_frame(input_path, snapshot)
    if meta_path:
        long = apply_metadata(long, load_metadata(meta_path))
    blank = long["artikelbezeichnung"].fillna("").astype(str).str.strip().eq("").mean()
    if blank > 0.5:
        print(f"  note {blank:.0%} of rows have no product name - the channel sheet has "
              f"#REF! there" + ("" if meta_path else "; pass --meta <retail export> to fill them in"))

    available = set(long["forecasted_date"].unique())
    missing = [t for t in new_months if t.to_datetime64() not in available]
    for t in missing:
        print(f"  warn {t:%b %Y} - no forecast columns for this month in the file")
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
        by_label = ", ".join(f"{lab}: {cnt}" for lab, cnt in part["label"].value_counts().sort_index().items())
        print(f"  add  {t:%b %Y} - {len(part):>5} rows, {part['sku'].nunique()} SKUs ({by_label})")

    out = result.copy()
    out["forecasted_date"] = out["forecasted_date"].dt.strftime("%Y-%m-%d")
    out["recorded_fetch_date"] = out["recorded_fetch_date"].dt.strftime("%Y-%m-%d")

    target_file = output_path or existing_path or Path("frozen_forecast.csv")
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
    p = argparse.ArgumentParser(description="Freeze the next N months of a forecast export.")
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="channel forecast export, FC by Channel (.csv/.xlsx)")
    p.add_argument("--meta", type=Path, default=None,
                   help="optional sheet to fill empty status/name/category from, by SKU")
    p.add_argument("-e", "--existing", type=Path, default=None, help="already frozen output file")
    p.add_argument("-o", "--output", type=Path, default=None, help="where to write the frozen file")
    p.add_argument("-m", "--file-month", default=None, help="override snapshot month, e.g. 2026-06")
    p.add_argument("-n", "--horizon", type=int, default=2, help="months to freeze per snapshot (default 2)")
    p.add_argument("--force", action="store_true", help="overwrite months that are already frozen")
    p.add_argument("--dry-run", action="store_true", help="print what would happen, write nothing")
    a = p.parse_args(argv)

    for path in (a.input, a.meta):
        if path is not None and not path.exists():
            print(f"input not found: {path}", file=sys.stderr)
            return 1

    file_month = None
    if a.file_month:
        file_month = pd.Timestamp(a.file_month).replace(day=1).normalize()

    freeze(a.input, a.existing, a.output, a.horizon, file_month, a.force, a.dry_run,
           meta_path=a.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())