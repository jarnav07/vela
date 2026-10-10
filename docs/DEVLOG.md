# Development Log

A chronological record of what was done, what went wrong and how it was fixed.
Design decisions are summarised in [DECISIONS.md](DECISIONS.md) (referenced as D*n*).

---

## Phase 0 — Environment and data access

**Problem: no market data access.** Yahoo Finance, Stooq, Kaggle, FRED, Wikipedia and HuggingFace all return
`403` from the sandbox's egress proxy. GitHub (git, raw content, release assets) and PyPI work.

- Searched for GitHub-hosted equity datasets. `AlanWangyl/NASDAQ_Historical_Daily_Prices_Dataset`
  turned out to hold only the download notebook (its data lives on Kaggle).
- Microsoft Qlib's US dataset is distributed as a GitHub **release asset** (`qlib_data_us_1d_latest.zip`,
  450 MB), which downloads fine. It covers 2000-01-03 → 2020-11-10 for ~9,000 symbols, plus
  `instruments/sp500.txt` with point-in-time membership intervals. **Adopted (D1).**
- All release tags (v0–v3) carry the same archive, so there is no newer US data from this source.

**Problem: PyTorch would not import.** PyPI's Linux wheel is a CUDA build and fails with
`libcudart.so.13: cannot open shared object file` when installed `--no-deps`. The CPU wheel index is blocked.
Fix: install with its NVIDIA runtime dependencies. It then runs on CPU (`torch.cuda.is_available() == False`) (D6).

**Note: GCS bucket.** The user pointed to the AERIS Google Cloud Storage bucket for large files. The bucket
(`gs://aeris-datasets-ml-aerofoil`) was found via the AERIS repo's `CLAUDE.md`, but the sandbox's GCS
credentials are rejected (`401 Invalid Credentials`). The cleaned dataset is ~150 MB and fully
reproducible from the public archive, so nothing depends on the bucket. `scripts/sync_gcs.sh` is
provided for when credentials are available.

## Phase 1 — Data layer and data quality

Wrote `vela/data/qlib_source.py`, a reader for Qlib's binary format: a float32 array whose first element is
the calendar offset. Symbol folders are lower-cased with an `_` prefix, and `BRK.B` becomes `_brk-b`.
`vela/data/panel.py` builds aligned `T × N` arrays and the membership matrix.

**Checks on the raw data**
- AAPL's adjusted closes and dollar volumes reconcile with known values (2020-11-10: ≈ $16bn traded).
- `^GSPC` close × the 1999-12-31 level (1469.25) gives 3545 on 2020-11-10, which matches the real index.
- SPY total-return CAGR 2000–2020 is 6.3% against 4.3% for the price index. The ~2% gap is dividends,
  so SPY is the benchmark used.
- 86 of 755 membership records have no price data (acquired or bankrupt names such as YHOO, CELG, LEH).
  This is residual survivorship bias, documented in D2.

**Problem: corrupt series.** Scanning member-days for |return| > 50% found 582 events. Many were garbage:
- CBE alternated between two price scales (−98%, +4,000%, −98% …) for years: ticker reuse.
- TIE jumped ×1,900 and back; HNZ alternated −66% / +194%; CHK compounded ×2–×25 jumps every quarter
  (broken dividend adjustment); BMC +347% with no reversal (an adjustment level shift).
- CEG's pre-2012 history is a different, illiquid security (57k shares/day for an S&P 500 company).

Others were real and must stay: AAPL −52% (2000-09-29), WMB −61% (2002), AIG −61% (2008-09-15),
financials in late 2008.

**Iterations on the cleaning rule (D3)**
1. *Volatility-scaled outliers*: abandoned. MRK's real −27% (Vioxx) is 36σ while DHR's +52% adjustment
   glitch is 53σ, so no threshold separates them.
2. *Neutralise any |log r| > log 3*: rejected. It removed AIG's real −67% open-to-open gap (2008-09-16) and
   dropped ABK (Ambac), whose glitches all fall *after* it left the index, which would have removed a
   genuine collapse from the sample.
3. **Final rule**: count ×3 / ÷3 moves **only inside the membership window**, and drop tickers with ≥ 2.
   8 dropped (CBE 352 glitches, TIE 31, CHK 14, HNZ 4, BMC 3, MEE 3, GR 3, TRB 2). AIG and PWR keep their
   single real events.

A 21-day median dollar-volume floor of $5M (`vela/universe.py`) also removes junk series like CEG's
without hindsight. Tradable names per day: ~290 (2000) → ~440 (2015) → ~500 (2020).
