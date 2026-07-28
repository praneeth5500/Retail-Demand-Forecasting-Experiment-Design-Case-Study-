"""Central configuration.

Every analytical decision that a reviewer might question lives here, in one
place, with a comment explaining WHY. This is the difference between a
notebook and a project.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "sales_data_2020_2025.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
EXTERNAL_DIR = ROOT / "data" / "external"
REPORTS_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Revenue definition
# ---------------------------------------------------------------------------
# The POS export mixes real revenue lines with derived/accounting lines.
# Summing the Amount column blindly is what produced the wrong headline
# numbers in v1 of this project.
#
# EXCLUDED, with reasons:
#   SalesTax  -> Sales tax collected on behalf of the state of Florida. It is
#                a liability, not revenue. Verified: it equals 7.02% (+/- 0.09pp)
#                of the sum of the taxable departments, i.e. it is DERIVED from
#                other rows in this same file. Including it double counts.
#   Tender    -> A single "Pump Test" line of -$11.70 (operational, not a sale).
#   Item Ctg  -> A stray header row that survived the export.
REVENUE_CATEGORIES = ["DeptSale", "FuelSale", "Lottery", "OtherIncome"]
EXCLUDED_CATEGORIES = ["SalesTax", "Tender", "Item Ctg"]

# Lottery is recorded as GROSS ticket sales; the store keeps only a ~5-6%
# commission. Fuel is high volume / low margin. Neither is wrong to include in
# a "sales" figure, but the write-up must say so. Set to True to report a
# merchandise-only view alongside the headline.
MERCHANDISE_ONLY_CATEGORIES = ["DeptSale"]

# ---------------------------------------------------------------------------
# Structural break
# ---------------------------------------------------------------------------
# On 2021-09-01 six departments appear at once (REG, PRE, MID, Soda, Tobacco,
# Fountain/Coffee become continuous). This is a POS/reporting change, not
# business growth. Data before this date measures a NARROWER product scope and
# is not comparable. All modeling starts here.
ANALYSIS_START = "2021-09-01"

# ---------------------------------------------------------------------------
# Forecasting setup
# ---------------------------------------------------------------------------
HORIZON = 30          # business ask: "next 30 days"
N_FOLDS = 6           # rolling-origin evaluation folds
FOLD_STEP = 30        # days between successive forecast origins
MIN_TRAIN_DAYS = 400  # need >364 to compute year-ago lags
SEASONAL_PERIOD = 7   # weekly seasonality dominates daily retail
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Experiment design (switchback) -- see EXPERIMENT_DESIGN.md
# ---------------------------------------------------------------------------
# This store has no customer identifiers and no loyalty program, so customers
# cannot be randomized. The only feasible design randomizes DAYS: the store is
# its own control on the days it is not treated ("switchback"). Everything in
# src/experiment.py is OFFLINE design work on historical observational data --
# no treatment was ever applied to this store.
#
# Primary metric: daily inside-merchandise revenue = sum of the DeptSale
# departments the store actually controls. Fuel is excluded (the store does not
# set fuel price) and lottery is excluded (recorded as gross ticket sales on a
# ~5-6% commission, so a dollar of lottery is not a dollar of margin-bearing
# sales). Air Vac is an OtherIncome line, not a DeptSale, and is excluded too.
INSIDE_MERCH_DEPARTMENTS = [
    "Cigarette", "Beer", "Soda", "Sales Taxable", "Sales Non-Tax",
    "Fountain/Coffee", "Autoparts", "Cigarette Carton", "Tobacco",
]
FUEL_DEPARTMENTS = ["REG", "PRE", "MID"]        # guardrail: fuel volume proxy
LOTTERY_DEPARTMENTS = ["Lotto Sales", "Scratchoff Sales"]

# Feasibility triage and variance estimates are computed on a RECENT window so
# they reflect the store as it is now, not the 2021-2022 post-break ramp. One
# whole year (52 weeks) ending at the last observation: long enough to estimate
# a daily standard deviation and a yearly seasonal shape, recent enough to be
# representative.
FEASIBILITY_WINDOW_WEEKS = 52

# Power / MDE assumptions, stated once so a reviewer can change them in one place.
ALPHA = 0.05          # two-sided significance
POWER = 0.80          # conventional target power
POWER_DURATIONS_WEEKS = [4, 6, 8, 12, 16]   # candidate switchback lengths
PLANNED_DURATION_WEEKS = 8                  # the length the A/A test validates

# A plausible upper bound on the lift a "discounted drink with fuel purchase"
# promo could realistically produce on a single inside-merchandise category.
# Used ONLY to flag departments whose minimum detectable effect is larger than
# any effect the intervention could plausibly generate -- i.e. not worth
# testing. It is a judgement call, stated here so it is not buried.
PLAUSIBLE_MAX_LIFT_PCT = 10.0

# A/A validation: how many random draw-assign-analyze iterations, and the seed.
AA_ITERATIONS = 2000
EXPERIMENT_SEED = 20210901   # distinct from RANDOM_SEED; the structural-break date
