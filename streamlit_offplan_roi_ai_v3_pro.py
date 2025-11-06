# 🏙️ Off-Plan ROI AI — Oracle Intelligence Cloud v3 (Pro+PDF+Tornado)
# One-file Streamlit app: regimes, correlated Monte-Carlo, discrete risks, offline mock data,
# Tornado sensitivity chart, and PDF investor report export.
# -------------------------------------------------------------------------
# Run locally:
#   pip install streamlit plotly pandas numpy reportlab
#   streamlit run streamlit_offplan_roi_ai_v3_pro.py
# -------------------------------------------------------------------------

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# PDF (ReportLab)
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Off-Plan ROI AI — v3 Pro+", page_icon="🏙️", layout="wide")

# ===============================
# 📊 Data models & utilities
# ===============================

@dataclass
class PaymentMilestone:
    milestone: str
    percent: float
    month: int
    flat_aed: float = 0.0

@dataclass
class ProjectInput:
    project_name: str
    developer: str
    unit_type: str
    base_price_aed: float
    size_sqft: float
    handover_month: int
    sale_month_from_start: int
    dld_fee_percent: float = 4.0
    agency_buy_percent: float = 2.0
    agency_sell_percent: float = 2.0
    other_buy_fees_percent: float = 0.0
    selling_costs_percent: float = 0.0
    eoi_aed: float = 0.0
    payment_plan: List[PaymentMilestone] = field(default_factory=list)
    appreciation_annual_percent: float = 8.0
    expected_rent_annual_aed: float = 0.0
    years_rented_post_handover: float = 0.0
    occupancy_percent: float = 90.0
    service_charges_aed_per_sqft_year: float = 0.0
    maintenance_percent_of_rent: float = 5.0
    furnishing_aed: float = 0.0
    use_mortgage: bool = False
    ltv_percent: float = 0.0
    mortgage_rate_annual_percent: float = 0.0
    mortgage_years: int = 25

@dataclass
class MarketRegimes:
    # annual means (percentage points) and vols (as % of mean) for each regime
    appr_mu_pp: Dict[str, float]             # {'bear':2, 'base':8, 'bull':14}
    appr_vol_pct: Dict[str, float]           # {'bear':0.35, ...}
    rent_mu_pct: Dict[str, float]            # rent scale drift per year (bear/base/bull)
    rent_vol_pct: Dict[str, float]
    weights: Dict[str, float]                # probabilities sum to 1

@dataclass
class RiskToggles:
    rho_price_rent: float = 0.45             # correlation of shocks
    handover_delay_mean_m: float = 2.0       # months
    handover_delay_sd_m: float = 1.0
    vacancy_days_per_year: int = 18
    rent_free_months_on_first_lease: float = 0.0
    refinance_prob: float = 0.15
    refinance_rate_delta_pp: float = -0.5    # drop after refi (pp)
    early_exit_prob: float = 0.10
    early_exit_month_shift: int = -6         # sell 6 months earlier
    svc_inflation_pct_pa: float = 4.0

def month_to_date(month_offset: int, start: Optional[datetime] = None) -> datetime:
    start = start or datetime.today()
    return start + timedelta(days=round(month_offset * 30.4375))

def ann_to_monthly(rate_annual: float) -> float:
    return (1 + rate_annual) ** (1/12) - 1

def amortized_payment(principal: float, annual_rate_percent: float, years: int) -> float:
    if principal <= 0:
        return 0.0
    r = ann_to_monthly(annual_rate_percent/100.0)
    n = years * 12
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

def npv_monthly(rate_annual: float, series: List[float]) -> float:
    r = (1 + rate_annual)**(1/12) - 1
    return sum(cf / ((1 + r) ** i) for i, cf in enumerate(series))

def irr_from_series(series: List[float]) -> float:
    # robust bisection for annual IRR using monthly discounting
    low, high = -0.99, 5.0
    for _ in range(120):
        mid = (low + high) / 2
        npv_mid = npv_monthly(mid, series)
        npv_low = npv_monthly(low, series)
        if npv_low * npv_mid > 0:
            low = mid
        else:
            high = mid
        if abs(npv_mid) < 1e-6:
            break
    return (low + high) / 2

# ===============================
# ⚙️ Core Engine
# ===============================

class OffPlanEngine:
    def __init__(self, p: ProjectInput, start_date: Optional[datetime] = None):
        self.p = p
        self.start_date = start_date or datetime.today()
        self.rows: List[Dict[str, Any]] = []
        self._build()

    def _add(self, month: int, desc: str, amt: float, eq: float = 0.0):
        self.rows.append({
            "month": month,
            "date": month_to_date(month, self.start_date),
            "description": desc,
            "cashflow_aed": amt,
            "equity_injected_aed": eq
        })

    def _build(self):
        p = self.p
        # EOI
        if p.eoi_aed > 0:
            self._add(0, "EOI", p.eoi_aed, eq=p.eoi_aed)

        # Buy fees at booking
        buy_fees = p.base_price_aed * (p.dld_fee_percent + p.agency_buy_percent + p.other_buy_fees_percent) / 100.0

        # Milestones
        for m in p.payment_plan:
            amt = m.flat_aed if m.flat_aed > 0 else p.base_price_aed * (m.percent / 100.0)
            if m.month == 0 and buy_fees > 0:
                self._add(m.month, f"{m.milestone} + Buy Fees", amt + buy_fees, eq=amt + buy_fees)
                buy_fees = 0.0
            else:
                self._add(m.month, m.milestone, amt, eq=amt)

        # Furnishing at handover
        if p.furnishing_aed > 0:
            self._add(p.handover_month, "Furnishing", p.furnishing_aed, eq=p.furnishing_aed)

        # Mortgage (optional; simple treatment)
        if p.use_mortgage and p.ltv_percent > 0:
            loan = p.base_price_aed * (p.ltv_percent / 100.0)
            self._add(p.handover_month, "Mortgage Proceeds (Loan)", -loan, eq=0.0)
            pay = amortized_payment(loan, p.mortgage_rate_annual_percent, p.mortgage_years)
            for i in range(p.mortgage_years * 12):
                self._add(p.handover_month + i, "Mortgage Payment", pay, eq=0.0)

        # Rental inflows
        if p.expected_rent_annual_aed > 0 and p.years_rented_post_handover > 0:
            months = int(round(p.years_rented_post_handover * 12))
            gross = p.expected_rent_annual_aed / 12.0
            occ = p.occupancy_percent / 100.0
            svc = (p.service_charges_aed_per_sqft_year * p.size_sqft) / 12.0
            maint = (p.maintenance_percent_of_rent / 100.0) * gross
            net = gross * occ - svc - maint
            for i in range(months):
                self._add(p.handover_month + i, "Net Rent Inflow", -max(0.0, net))

        # Sale proceeds
        years = max(0.0, p.sale_month_from_start / 12.0)
        sale_price = p.base_price_aed * ((1 + p.appreciation_annual_percent / 100.0) ** years)
        sell_costs = sale_price * (p.agency_sell_percent + p.selling_costs_percent) / 100.0
        self._add(p.sale_month_from_start, "Sale Proceeds (net)", -(sale_price - sell_costs))

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values(by=["month","description"]).reset_index(drop=True)

    def totals(self) -> Dict[str, float]:
        df = self.df()
        out = df.loc[df["cashflow_aed"] > 0, "cashflow_aed"].sum()
        inflow = -df.loc[df["cashflow_aed"] < 0, "cashflow_aed"].sum()
        eq = df["equity_injected_aed"].sum()
        return {"outflows": float(out), "inflows": float(inflow), "profit": float(inflow - out), "equity": float(eq)}

    def metrics(self) -> Dict[str, float]:
        t = self.totals()
        out = t["outflows"]; eq = max(1e-9, t["equity"])
        df = self.df()
        rent_in = -df.loc[df["description"] == "Net Rent Inflow", "cashflow_aed"].sum()
        sale_in = -df.loc[df["description"] == "Sale Proceeds (net)", "cashflow_aed"].sum()
        cap_profit = sale_in - out
        rent_profit = rent_in
        series = [-r for r in df["cashflow_aed"].tolist()]
        irr = irr_from_series(series) if len(series) >= 2 else float('nan')
        return {
            "Capital ROI %": (cap_profit/out*100.0) if out>0 else float('nan'),
            "Rental ROI %": (rent_profit/out*100.0) if out>0 else float('nan'),
            "Total ROI %": ((cap_profit+rent_profit)/out*100.0) if out>0 else float('nan'),
            "ROE %": (t["profit"]/eq*100.0) if eq>0 else float('nan'),
            "IRR (annual) %": irr*100 if not math.isnan(irr) else float('nan')
        }

# ===============================
# 🔧 Sensitivity grid (bear/base/bull)
# ===============================

def sensitivity_grid(p: ProjectInput, appr_shifts_pp=(-5,0,5), rent_shifts_pct=(-0.10,0.0,0.10)) -> pd.DataFrame:
    rows = []
    for a in appr_shifts_pp:
        for r in rent_shifts_pct:
            p2 = ProjectInput(**{**p.__dict__})
            p2.appreciation_annual_percent = max(0.0, p.appreciation_annual_percent + a)
            p2.expected_rent_annual_aed = max(0.0, p.expected_rent_annual_aed * (1 + r))
            eng = OffPlanEngine(p2)
            m = eng.metrics()
            rows.append({
                "Scenario": f"appr {a:+}pp | rent {int(r*100):+}%",
                "Capital ROI %": m["Capital ROI %"],
                "Rental ROI %": m["Rental ROI %"],
                "Total ROI %": m["Total ROI %"],
                "ROE %": m["ROE %"],
                "IRR (annual) %": m["IRR (annual) %"]
            })
    return pd.DataFrame(rows)

# ===============================
# 📡 Offline evidence mocks
# ===============================

def mock_dld_comps(area:str, project:str):
    seed = abs(hash((area, project))) % 10_000
    random.seed(seed)
    avg_psf = random.randint(1700, 3200)
    mu_app = random.uniform(5.5, 11.0)     # annual pp
    return {"avg_psf": avg_psf, "appreciation_mu_pp": mu_app}

def mock_rera_pf(area:str, unit_type:str):
    seed = abs(hash((area, unit_type))) % 10_000
    random.seed(seed)
    rent = random.randint(70000, 220000)
    occ = random.uniform(0.86, 0.95)
    return {"annual_rent": rent, "occupancy": occ}

def mock_mortgage_quotes(base_price:float):
    seed = abs(hash(base_price)) % 10_000
    random.seed(seed)
    ltv = random.choice([50, 60, 70, 75, 80])
    rate = random.uniform(4.5, 6.5)
    bank_fees = base_price * random.uniform(0.003, 0.007)  # 0.3–0.7%
    return {"ltv": ltv, "rate": rate, "bank_fees": bank_fees}

def mock_service_charges(project:str, sqft:float):
    seed = abs(hash((project, sqft))) % 10_000
    random.seed(seed)
    svc_psf = random.uniform(18, 32)  # AED/sqft/year
    return {"svc_psf": svc_psf}

def build_regimes_from_data(dld_mu_pp: float) -> MarketRegimes:
    # tie base regime to DLD-derived appreciation mean
    bear_mu = max(0.0, dld_mu_pp - 6)
    base_mu = dld_mu_pp
    bull_mu = dld_mu_pp + 6
    return MarketRegimes(
        appr_mu_pp={"bear": bear_mu, "base": base_mu, "bull": bull_mu},
        appr_vol_pct={"bear": 0.35, "base": 0.25, "bull": 0.30},
        rent_mu_pct={"bear": -0.05, "base": 0.00, "bull": 0.05},
        rent_vol_pct={"bear": 0.20, "base": 0.12, "bull": 0.18},
        weights={"bear": 0.25, "base": 0.55, "bull": 0.20},
    )

# ===============================
# 🧠 Correlated Monte-Carlo (regimes + discrete risks)
# ===============================

def simulate_roi_distribution_advanced(
    p: ProjectInput,
    regimes: MarketRegimes,
    risks: RiskToggles,
    n_iter: int = 10000
) -> pd.DataFrame:
    rho = max(min(risks.rho_price_rent, 0.99), -0.99)
    cov = np.array([[1.0, rho],[rho,1.0]])
    L = np.linalg.cholesky(cov)

    rows = []
    reg_names = list(regimes.weights.keys())
    reg_probs = np.array([regimes.weights[k] for k in reg_names])
    reg_probs = reg_probs / reg_probs.sum()

    for _ in range(n_iter):
        # draw regime
        regime = np.random.choice(reg_names, p=reg_probs)

        # correlated shocks
        z = np.dot(L, np.random.normal(size=2))

        # appreciation (annual pp)
        appr_mu = regimes.appr_mu_pp[regime]
        appr_vol = max(0.0001, abs(appr_mu) * regimes.appr_vol_pct[regime])
        appr_draw = max(0.0, appr_mu + z[0] * appr_vol)

        # rent scale
        rent_mu = regimes.rent_mu_pct[regime]
        rent_vol = max(0.0001, max(0.02, abs(rent_mu)) * regimes.rent_vol_pct[regime])
        rent_scale = 1.0 + (rent_mu + z[1]*rent_vol)
        rent_scale = max(0.6, min(1.4, rent_scale))

        # clone and apply draws
        p2 = ProjectInput(**{**p.__dict__})
        p2.appreciation_annual_percent = appr_draw
        p2.expected_rent_annual_aed = max(0.0, p.expected_rent_annual_aed * rent_scale)
        # occupancy from vacancy days
        occ_days = max(0, int(risks.vacancy_days_per_year))
        p2.occupancy_percent = max(0, min(100, (365 - occ_days) / 365 * 100))

        # discrete risks
        delay = max(0, int(np.random.normal(risks.handover_delay_mean_m, risks.handover_delay_sd_m)))
        p2.handover_month = p.handover_month + delay

        if risks.rent_free_months_on_first_lease > 0 and p.years_rented_post_handover>0:
            rf = int(risks.rent_free_months_on_first_lease)
            p2.years_rented_post_handover = max(0.0, p.years_rented_post_handover - rf/12.0)

        refi = (np.random.rand() < risks.refinance_prob)
        if refi and p.use_mortgage:
            p2.mortgage_rate_annual_percent = max(0.0, p.mortgage_rate_annual_percent + risks.refinance_rate_delta_pp)

        early = (np.random.rand() < risks.early_exit_prob)
        if early:
            p2.sale_month_from_start = max(1, p.sale_month_from_start + risks.early_exit_month_shift)

        # service-charge inflation (terminal uplift approximation)
        if p2.service_charges_aed_per_sqft_year > 0 and p2.years_rented_post_handover>0:
            years = p2.years_rented_post_handover
            infl = (1 + risks.svc_inflation_pct_pa/100.0) ** years
            p2.service_charges_aed_per_sqft_year *= infl

        # evaluate
        e = OffPlanEngine(p2)
        m = e.metrics()
        rows.append({
            "regime": regime,
            "Total ROI %": m["Total ROI %"],
            "Capital ROI %": m["Capital ROI %"],
            "Rental ROI %": m["Rental ROI %"],
            "ROE %": m["ROE %"],
        })
    return pd.DataFrame(rows)

# Downside helpers
def pctl(arr, q): 
    return float(np.percentile(arr, q))

def cvar(arr, tail=10):
    cutoff = np.percentile(arr, tail)
    return float(np.mean(arr[arr <= cutoff]))

# ===============================
# 🧱 Developer templates (for quick start)
# ===============================

DEV_TEMPLATES = {
    "Emaar":  [PaymentMilestone("Booking",10,0), PaymentMilestone("During Construction",70,24), PaymentMilestone("Handover",20,36)],
    "Damac":  [PaymentMilestone("Booking",20,0), PaymentMilestone("During Construction",50,30), PaymentMilestone("Handover",30,42)],
    "Sobha":  [PaymentMilestone("Booking",10,0), PaymentMilestone("During Construction",60,36), PaymentMilestone("Handover",30,48)],
    "Nakheel":[PaymentMilestone("Booking",15,0), PaymentMilestone("During Construction",55,24), PaymentMilestone("Handover",30,36)],
    "Ellington":[PaymentMilestone("Booking",20,0), PaymentMilestone("During Construction",50,18), PaymentMilestone("Handover",30,30)],
}

# ===============================
# 🧱 Streamlit UI
# ===============================

st.title("🏙️ Off-Plan ROI AI — Oracle Intelligence v3 Pro+")
st.caption("Evidence-based simulator with regimes, correlated Monte-Carlo, discrete risks, offline mock data, Tornado chart, and PDF export.")

# Sidebar — base project inputs
with st.sidebar:
    st.header("📈 Project Setup")
    dev_choice = st.selectbox("Developer", list(DEV_TEMPLATES.keys()), index=0)
    name = st.text_input("Project Name", f"{dev_choice} Flagship")
    developer = st.text_input("Developer (editable)", dev_choice)
    unit_type = st.text_input("Unit Type", "2BR")

    base_price = st.number_input("Base Price (AED)", min_value=0.0, value=2_000_000.0, step=10_000.0, format="%.2f")
    size_sqft = st.number_input("Size (sqft)", min_value=0.0, value=1200.0, step=10.0)
    handover_m = st.number_input("Handover (months)", min_value=0, value=36, step=1)
    sale_m = st.number_input("Planned Sale Month", min_value=0, value=48, step=1)

    st.subheader("Fees & EOI")
    dld = st.number_input("DLD %", min_value=0.0, value=4.0)
    agency_buy = st.number_input("Agency Buy %", min_value=0.0, value=2.0)
    agency_sell = st.number_input("Agency Sell %", min_value=0.0, value=2.0)
    other_buy = st.number_input("Other Buy Fees %", min_value=0.0, value=0.0)
    exit_cost = st.number_input("Other Exit Costs %", min_value=0.0, value=0.0)
    eoi = st.number_input("EOI (AED)", min_value=0.0, value=10_000.0, step=1_000.0)

    st.subheader("Payment Plan (template)")
    m1, m2, m3 = DEV_TEMPLATES[dev_choice]
    m1_name = st.text_input("M1 Name", m1.milestone)
    m1_pct = st.number_input("M1 % of base", min_value=0.0, value=float(m1.percent))
    m1_month = st.number_input("M1 Month", min_value=0, value=int(m1.month))
    m2_name = st.text_input("M2 Name", m2.milestone)
    m2_pct = st.number_input("M2 % of base", min_value=0.0, value=float(m2.percent))
    m2_month = st.number_input("M2 Month", min_value=0, value=int(m2.month))
    m3_name = st.text_input("M3 Name", m3.milestone)
    m3_pct = st.number_input("M3 % of base", min_value=0.0, value=float(m3.percent))
    m3_month = st.number_input("M3 Month", min_value=0, value=int(m3.month))

    st.subheader("Market (base)")
    appr = st.slider("Appreciation % (annual, base)", min_value=0.0, max_value=20.0, value=8.0, step=0.5)
    rent = st.number_input("Gross Annual Rent (AED)", min_value=0.0, value=120_000.0, step=5_000.0)
    years_rent = st.number_input("Years Rented Post-Handover", min_value=0.0, value=2.0, step=0.5)
    occ = st.slider("Occupancy %", min_value=50, max_value=100, value=90, step=1)
    svc = st.number_input("Service Charges (AED/sqft/year)", min_value=0.0, value=24.0, step=1.0)
    maint = st.number_input("Maintenance % of Rent", min_value=0.0, value=5.0, step=0.5)
    furnish = st.number_input("Furnishing (AED at Handover)", min_value=0.0, value=30_000.0, step=1_000.0)

    st.subheader("Financing (optional)")
    use_mort = st.checkbox("Use mortgage?", value=False)
    ltv = st.number_input("LTV %", min_value=0.0, max_value=80.0, value=50.0) if use_mort else 0.0
    mort_rate = st.number_input("Mortgage Rate % (annual)", min_value=0.0, max_value=12.0, value=5.0) if use_mort else 0.0
    mort_years = st.number_input("Mortgage Term (years)", min_value=1, max_value=30, value=25) if use_mort else 25

    st.subheader("Simulation Settings")
    sims = st.number_input("Monte Carlo runs", min_value=1000, max_value=50000, value=10000, step=1000)

    st.markdown("---")
    st.subheader("Market Data (offline mocks)")
    area = st.text_input("Area", "Dubai Creek Harbour")
    project_name_input = st.text_input("Project (for comps)", name)
    unit_type_ui = st.text_input("Unit Type (for rent comps)", unit_type)
    use_mocks = st.checkbox("Auto-fill from mocks", value=True)

    st.subheader("Regime Weights")
    bear_w = st.slider("Bear", 0.0, 1.0, 0.25, 0.05)
    base_w = st.slider("Base", 0.0, 1.0, 0.55, 0.05)
    bull_w = st.slider("Bull", 0.0, 1.0, 0.20, 0.05)
    total_w = bear_w + base_w + bull_w
    if total_w == 0: bear_w, base_w, bull_w = 0.25, 0.55, 0.20

    st.subheader("Risk Toggles")
    rho = st.slider("Correlation (price ↔ rent)", -0.9, 0.9, 0.45, 0.05)
    delay_mean = st.slider("Handover delay mean (months)", 0, 12, 2)
    delay_sd = st.slider("Handover delay sd (months)", 0, 6, 1)
    vac_days = st.slider("Vacancy days / year", 0, 60, 18)
    rent_free = st.slider("Rent-free months (first lease)", 0.0, 3.0, 0.0, 0.5)
    refi_prob = st.slider("Refinance probability", 0.0, 1.0, 0.15, 0.05)
    refi_delta = st.slider("Refi rate delta (pp)", -2.0, 2.0, -0.5, 0.1)
    early_prob = st.slider("Early exit probability", 0.0, 1.0, 0.10, 0.05)
    early_shift = st.slider("Early exit month shift", -12, 0, -6)
    svc_infl = st.slider("Service charge inflation %/yr", 0.0, 10.0, 4.0, 0.5)

# Build project instance from sidebar
project = ProjectInput(
    project_name=name, developer=developer, unit_type=unit_type,
    base_price_aed=base_price, size_sqft=size_sqft,
    handover_month=handover_m, sale_month_from_start=sale_m,
    dld_fee_percent=dld, agency_buy_percent=agency_buy, agency_sell_percent=agency_sell,
    other_buy_fees_percent=other_buy, selling_costs_percent=exit_cost, eoi_aed=eoi,
    payment_plan=[
        PaymentMilestone(m1_name, m1_pct, m1_month),
        PaymentMilestone(m2_name, m2_pct, m2_month),
        PaymentMilestone(m3_name, m3_pct, m3_month),
    ],
    appreciation_annual_percent=appr, expected_rent_annual_aed=rent, years_rented_post_handover=years_rent,
    occupancy_percent=float(occ), service_charges_aed_per_sqft_year=svc, maintenance_percent_of_rent=maint,
    furnishing_aed=furnish, use_mortgage=use_mort, ltv_percent=ltv, mortgage_rate_annual_percent=mort_rate, mortgage_years=mort_years
)

# ===============================
# 🔎 Apply offline evidence mocks (optional)
# ===============================

if use_mocks:
    dld_data = mock_dld_comps(area, project_name_input)
    rent_data = mock_rera_pf(area, unit_type_ui)
    mort_data = mock_mortgage_quotes(base_price)
    svc_data = mock_service_charges(project_name_input, size_sqft)

    # non-destructive override of key assumptions (you can still tweak in UI)
    project.expected_rent_annual_aed = rent_data["annual_rent"]
    project.occupancy_percent = rent_data["occupancy"] * 100
    project.service_charges_aed_per_sqft_year = svc_data["svc_psf"]

    regimes = build_regimes_from_data(dld_data["appreciation_mu_pp"])
else:
    regimes = build_regimes_from_data(project.appreciation_annual_percent)

# normalize regime weights from sliders
regimes.weights = {"bear": bear_w/total_w, "base": base_w/total_w, "bull": bull_w/total_w}

risks = RiskToggles(
    rho_price_rent=rho,
    handover_delay_mean_m=delay_mean,
    handover_delay_sd_m=delay_sd,
    vacancy_days_per_year=vac_days,
    rent_free_months_on_first_lease=rent_free,
    refinance_prob=refi_prob,
    refinance_rate_delta_pp=refi_delta,
    early_exit_prob=early_prob,
    early_exit_month_shift=early_shift,
    svc_inflation_pct_pa=svc_infl
)

# ===============================
# 📊 Base metrics (deterministic)
# ===============================

engine = OffPlanEngine(project)
df_base = engine.df()
tot = engine.totals()
met = engine.metrics()

st.subheader("📌 Base Case Metrics (deterministic)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Outflows (AED)", f"{tot['outflows']:,.0f}")
k2.metric("Total Inflows (AED)", f"{tot['inflows']:,.0f}")
k3.metric("Net Profit (AED)", f"{tot['profit']:,.0f}")
k4.metric("ROE %", f"{met['ROE %']:.2f}%")

m1, m2, m3 = st.columns(3)
m1.metric("Capital ROI %", f"{met['Capital ROI %']:.2f}%")
m2.metric("Rental ROI %", f"{met['Rental ROI %']:.2f}%")
m3.metric("Total ROI %", f"{met['Total ROI %']:.2f}%")

# Charts — cashflow & equity
st.subheader("💰 Cashflow Timeline (Base Case)")
cf_fig = px.bar(df_base, x="date", y="cashflow_aed", color="description",
                title="Monthly Cashflows (Outflows + / Inflows -)")
st.plotly_chart(cf_fig, use_container_width=True)

st.subheader("📈 Equity Accumulation (Cumulative Cashflow)")
df_sorted = df_base.sort_values("date").copy()
df_sorted["cum_cash"] = df_sorted["cashflow_aed"].cumsum()
eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=df_sorted["date"], y=df_sorted["cum_cash"], mode="lines", name="Cumulative Cashflow"))
eq_fig.update_layout(xaxis_title="Date", yaxis_title="AED")
st.plotly_chart(eq_fig, use_container_width=True)

# ===============================
# 🧪 Sensitivity Grid (Bear / Base / Bull)
# ===============================

st.subheader("🧪 Sensitivity Grid (Bear / Base / Bull)")
sens_df = sensitivity_grid(project)
st.dataframe(sens_df, use_container_width=True)

# ===============================
# 🌪️ Tornado Sensitivity (factor impact on Total ROI)
# ===============================

def tornado_analysis(p: ProjectInput) -> pd.DataFrame:
    """One-at-a-time sensitivity around current inputs; returns impact range for Total ROI %."""
    base = OffPlanEngine(p).metrics()["Total ROI %"]
    tests = [
        ("Appreciation (pp)", "appr_pp", -2.0, +2.0),
        ("Annual Rent (±10%)", "rent_pct", -0.10, +0.10),
        ("Occupancy (pp)", "occ_pp", -5.0, +5.0),
        ("Service Charges (AED/sqft)", "svc_psf", -3.0, +3.0),
        ("Maintenance (% rent)", "maint_pp", -2.0, +2.0),
        ("Selling Costs (pp)", "sell_pp", -1.0, +1.0),
        ("Handover Delay (months)", "delay_m", +0.0, +3.0),  # only upside delay risk
    ]
    rows = []
    for label, key, low, high in tests:
        p_low = ProjectInput(**{**p.__dict__})
        p_high = ProjectInput(**{**p.__dict__})
        if key == "appr_pp":
            p_low.appreciation_annual_percent = max(0, p.appreciation_annual_percent + low)
            p_high.appreciation_annual_percent = p.appreciation_annual_percent + high
        elif key == "rent_pct":
            p_low.expected_rent_annual_aed = max(0, p.expected_rent_annual_aed * (1+low))
            p_high.expected_rent_annual_aed = max(0, p.expected_rent_annual_aed * (1+high))
        elif key == "occ_pp":
            p_low.occupancy_percent = max(0, min(100, p.occupancy_percent + low))
            p_high.occupancy_percent = max(0, min(100, p.occupancy_percent + high))
        elif key == "svc_psf":
            p_low.service_charges_aed_per_sqft_year = max(0, p.service_charges_aed_per_sqft_year + low)
            p_high.service_charges_aed_per_sqft_year = max(0, p.service_charges_aed_per_sqft_year + high)
        elif key == "maint_pp":
            p_low.maintenance_percent_of_rent = max(0, p.maintenance_percent_of_rent + low)
            p_high.maintenance_percent_of_rent = max(0, p.maintenance_percent_of_rent + high)
        elif key == "sell_pp":
            p_low.selling_costs_percent = max(0, p.selling_costs_percent + low)
            p_high.selling_costs_percent = max(0, p.selling_costs_percent + high)
        elif key == "delay_m":
            p_low.handover_month = p.handover_month + int(abs(low))  # low here is 0
            p_high.handover_month = p.handover_month + int(abs(high))
        # evaluate
        low_roi = OffPlanEngine(p_low).metrics()["Total ROI %"]
        high_roi = OffPlanEngine(p_high).metrics()["Total ROI %"]
        lo, hi = sorted([low_roi, high_roi])
        impact = hi - lo
        rows.append({
            "Factor": label,
            "Low ROI %": lo,
            "High ROI %": hi,
            "Impact (pp)": impact,
            "Base ROI %": base
        })
    df = pd.DataFrame(rows).sort_values("Impact (pp)", ascending=True)
    return df

st.subheader("🌪️ Tornado Sensitivity — Factor Impact on Total ROI %")
tornado_df = tornado_analysis(project)
tornado_fig = go.Figure()
tornado_fig.add_trace(go.Bar(
    x=tornado_df["Impact (pp)"],
    y=tornado_df["Factor"],
    orientation="h",
    hovertemplate="Impact: %{x:.2f} pp<extra></extra>"
))
tornado_fig.update_layout(title="Tornado Chart (larger bar = bigger effect on Total ROI %)",
                          xaxis_title="Impact (percentage points)", yaxis_title="")
st.plotly_chart(tornado_fig, use_container_width=True)
st.dataframe(tornado_df.reset_index(drop=True), use_container_width=True)

# ===============================
# 🧠 Advanced Simulation (regimes + risks)
# ===============================

st.subheader("🧠 Monte-Carlo (Regimes + Correlation + Discrete Risks)")
with st.spinner("Running correlated Monte-Carlo with regimes & risks ..."):
    sim_df = simulate_roi_distribution_advanced(project, regimes, risks, n_iter=int(sims))

arr = sim_df["Total ROI %"].dropna().to_numpy()
summary = {
    "Mean ROI %": float(np.mean(arr)) if arr.size else float("nan"),
    "Median ROI %": float(np.median(arr)) if arr.size else float("nan"),
    "P50 ROI %": pctl(arr, 50) if arr.size else float("nan"),
    "P90 ROI % (downside)": pctl(arr, 10) if arr.size else float("nan"),
    "CVaR (worst 10%) %": cvar(arr, 10) if arr.size else float("nan"),
}

c1, c2 = st.columns([2,1])
with c1:
    st.plotly_chart(px.histogram(sim_df, x="Total ROI %", nbins=50, title="Simulated Total ROI %"), use_container_width=True)
    st.plotly_chart(px.box(sim_df, y="Total ROI %", points=False, title="ROI Spread (Boxplot)"), use_container_width=True)
with c2:
    st.write("**Downside & Summary Metrics**")
    st.json(summary)
    regime_share = sim_df["regime"].value_counts(normalize=True).rename("share").reset_index(names="regime")
    st.plotly_chart(px.pie(regime_share, names="regime", values="share", title="Regime Mix (simulated)"), use_container_width=True)

# ===============================
# 🧪 Backtest & calibration (demo)
# ===============================

with st.expander("Backtest & Calibration (offline demo)"):
    if arr.size:
        realized = arr + np.random.normal(0, np.std(arr)*0.2, size=len(arr))
        pred_std = float(np.std(arr)); real_std = float(np.std(realized))
        scale = 1.0 if pred_std==0 else max(0.5, min(2.0, real_std / pred_std))
        st.write({"pred_std": pred_std, "real_std": real_std, "vol_scale": scale})
        st.caption("Replace synthetic 'realized' with your historical ROI per project to tune distributions.")
    else:
        st.info("Run the simulation first.")

# ===============================
# 📄 PDF investor report
# ===============================

def build_pdf(project: ProjectInput, base_metrics: Dict[str,float], mc_summary: Dict[str,float]) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"<b>{project.project_name}</b>", styles["Title"]))
    story.append(Paragraph(f"Developer: {project.developer} | Unit: {project.unit_type}", styles["Normal"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Base metrics
    story.append(Paragraph("<b>Base Metrics</b>", styles["Heading2"]))
    base_items = [
        ["Capital ROI %", f"{base_metrics.get('Capital ROI %', float('nan')):,.2f}"],
        ["Rental ROI %", f"{base_metrics.get('Rental ROI %', float('nan')):,.2f}"],
        ["Total ROI %", f"{base_metrics.get('Total ROI %', float('nan')):,.2f}"],
        ["ROE %", f"{base_metrics.get('ROE %', float('nan')):,.2f}"],
        ["IRR (annual) %", f"{base_metrics.get('IRR (annual) %', float('nan')):,.2f}"],
    ]
    t1 = Table(base_items, colWidths=[8*cm, 4*cm])
    t1.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey),
                            ('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),
                            ('FONT',(0,0),(-1,0),'Helvetica-Bold')]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # Monte Carlo summary
    story.append(Paragraph("<b>Monte-Carlo Summary</b>", styles["Heading2"]))
    mc_items = [[k, f"{v:,.2f}"] for k,v in mc_summary.items()]
    t2 = Table(mc_items, colWidths=[8*cm, 4*cm])
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("Notes: This model uses regime-based, correlated simulations with discrete risk events. "
                           "All results are estimates and not financial advice.", styles["Italic"]))
    doc.build(story)
    buf.seek(0)
    return buf.getvalue()

st.subheader("📄 Export")
col_pdf, col_csv = st.columns([1,1])
with col_pdf:
    pdf_bytes = build_pdf(project, met, summary)
    st.download_button("Download Investor PDF", data=pdf_bytes,
                       file_name=f"{project.project_name.replace(' ','_')}_Investor_Report.pdf",
                       mime="application/pdf")
with col_csv:
    cf_csv = df_base.to_csv(index=False).encode("utf-8")
    st.download_button("Download Base Cashflow CSV", cf_csv, file_name="cashflow_timeline_base.csv", mime="text/csv")

# Simulation CSV (full)
sim_csv = sim_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Simulation Results CSV", sim_csv, file_name="simulation_results.csv", mime="text/csv")

st.caption("© 2025 Oracle Intelligence — Prototype build for Etibar. All results are estimates and not financial advice.")
