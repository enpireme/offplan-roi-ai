# 🏙️ Off-Plan ROI AI — Oracle Intelligence Cloud v2.0
# Streamlit Dashboard — created by Etibar & ChatGPT (GPT-5)
# -----------------------------------------------------------
# Run locally with:
#   pip install streamlit plotly pandas numpy
#   streamlit run streamlit_offplan_roi_ai_v2_0.py
# -----------------------------------------------------------

import math
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Off-Plan ROI AI — v2.0", page_icon="🏙️", layout="wide")

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
    low, high = -0.99, 5.0
    for _ in range(100):
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
        if p.eoi_aed > 0:
            self._add(0, "EOI", p.eoi_aed, eq=p.eoi_aed)

        buy_fees = p.base_price_aed * (p.dld_fee_percent + p.agency_buy_percent + p.other_buy_fees_percent) / 100.0

        for m in p.payment_plan:
            amt = m.flat_aed if m.flat_aed > 0 else p.base_price_aed * (m.percent / 100.0)
            if m.month == 0 and buy_fees > 0:
                self._add(m.month, f"{m.milestone} + Buy Fees", amt + buy_fees, eq=amt + buy_fees)
                buy_fees = 0.0
            else:
                self._add(m.month, m.milestone, amt, eq=amt)

        if p.furnishing_aed > 0:
            self._add(p.handover_month, "Furnishing", p.furnishing_aed, eq=p.furnishing_aed)

        if p.expected_rent_annual_aed > 0 and p.years_rented_post_handover > 0:
            months = int(round(p.years_rented_post_handover * 12))
            gross = p.expected_rent_annual_aed / 12.0
            occ = p.occupancy_percent / 100.0
            svc = (p.service_charges_aed_per_sqft_year * p.size_sqft) / 12.0
            maint = (p.maintenance_percent_of_rent / 100.0) * gross
            net = gross * occ - svc - maint
            for i in range(months):
                self._add(p.handover_month + i, "Net Rent Inflow", -max(0.0, net))

        years = max(0.0, p.sale_month_from_start / 12.0)
        sale_price = p.base_price_aed * ((1 + p.appreciation_annual_percent / 100.0) ** years)
        sell_costs = sale_price * (p.agency_sell_percent + p.selling_costs_percent) / 100.0
        self._add(p.sale_month_from_start, "Sale Proceeds (net)", -(sale_price - sell_costs))

    def df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows).sort_values(by=["month"]).reset_index(drop=True)

    def totals(self) -> Dict[str, float]:
        df = self.df()
        out = df.loc[df["cashflow_aed"] > 0, "cashflow_aed"].sum()
        inflow = -df.loc[df["cashflow_aed"] < 0, "cashflow_aed"].sum()
        eq = df["equity_injected_aed"].sum()
        return {"outflows": out, "inflows": inflow, "profit": inflow - out, "equity": eq}

    def metrics(self) -> Dict[str, float]:
        t = self.totals()
        out = t["outflows"]; eq = max(1e-9, t["equity"])
        df = self.df()
        rent_in = -df.loc[df["description"] == "Net Rent Inflow", "cashflow_aed"].sum()
        sale_in = -df.loc[df["description"] == "Sale Proceeds (net)", "cashflow_aed"].sum()
        cap_profit = sale_in - out; rent_profit = rent_in
        irr_series = [-r for r in df["cashflow_aed"].tolist()]
        irr = irr_from_series(irr_series)
        return {
            "Capital ROI %": (cap_profit/out*100) if out>0 else float('nan'),
            "Rental ROI %": (rent_profit/out*100) if out>0 else float('nan'),
            "Total ROI %": ((cap_profit+rent_profit)/out*100) if out>0 else float('nan'),
            "ROE %": (t["profit"]/eq*100) if eq>0 else float('nan'),
            "IRR (annual) %": irr*100
        }

# ===============================
# 🧠 Simulation & Visualization
# ===============================

# === PATCH: correlated Monte-Carlo with regimes & risks ===
def simulate_roi_distribution_advanced(
    p: ProjectInput,
    regimes: MarketRegimes,
    risks: RiskToggles,
    n_iter: int = 10000
) -> pd.DataFrame:
    # correlation matrix for price & rent growth shocks
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

        # draw correlated normals
        z = np.dot(L, np.random.normal(size=2))
        # appreciation mean & vol for regime
        appr_mu = regimes.appr_mu_pp[regime]
        appr_vol = max(0.0001, abs(appr_mu) * regimes.appr_vol_pct[regime])
        appr_draw = max(0.0, appr_mu + z[0] * appr_vol)  # annual pp

        # rent scale (%) for regime
        rent_mu = regimes.rent_mu_pct[regime]
        rent_vol = max(0.0001, max(0.02, abs(rent_mu)) * regimes.rent_vol_pct[regime])
        rent_scale = 1.0 + (rent_mu + z[1]*rent_vol)     # multiplicative to base rent
        rent_scale = max(0.6, min(1.4, rent_scale))

        # clone project with draws
        p2 = ProjectInput(**{**p.__dict__})
        p2.appreciation_annual_percent = appr_draw
        p2.expected_rent_annual_aed = max(0.0, p.expected_rent_annual_aed * rent_scale)
        # occupancy adjust via vacancy days
        occ_days = max(0, int(risks.vacancy_days_per_year))
        p2.occupancy_percent = max(0, min(100, (365 - occ_days) / 365 * 100))

        # discrete risks
        # handover delay
        delay = max(0, int(np.random.normal(risks.handover_delay_mean_m, risks.handover_delay_sd_m)))
        p2.handover_month = p.handover_month + delay
        # rent-free on first lease
        if risks.rent_free_months_on_first_lease > 0:
            rf = int(risks.rent_free_months_on_first_lease)
            # implement by pushing rent start
            p2.years_rented_post_handover = max(0.0, p.years_rented_post_handover - rf/12.0)

        # refinance chance
        refi = (np.random.rand() < risks.refinance_prob)
        if refi and p.use_mortgage:
            p2.mortgage_rate_annual_percent = max(0.0, p.mortgage_rate_annual_percent + risks.refinance_rate_delta_pp)

        # early exit chance
        early = (np.random.rand() < risks.early_exit_prob)
        if early:
            p2.sale_month_from_start = max(1, p.sale_month_from_start + risks.early_exit_month_shift)

        # service charge inflation (approx)
        if p2.service_charges_aed_per_sqft_year > 0 and p2.years_rented_post_handover>0:
            years = p2.years_rented_post_handover
            infl = (1 + risks.svc_inflation_pct_pa/100.0) ** years
            p2.service_charges_aed_per_sqft_year *= infl  # simple terminal uplift

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
# === PATCH: downside metrics ===
def pctl(arr, q): return float(np.percentile(arr, q))
def cvar(arr, tail=10):
    cutoff = np.percentile(arr, tail)
    return float(np.mean(arr[arr <= cutoff]))

# ===============================
# 🧱 Streamlit UI
# ===============================

st.title("🏙️ Off-Plan ROI AI — Oracle Intelligence v2.0")
st.caption("Dubai Off-Plan Investment Simulator (ROI • ROE • IRR • Monte Carlo)")

# Sidebar Inputs
with st.sidebar:
    st.header("📈 Project Setup")
    dev = st.selectbox("Developer", ["Emaar", "Damac", "Sobha", "Nakheel", "Ellington"])
    name = st.text_input("Project Name", f"{dev} Flagship")
    price = st.number_input("Base Price (AED)", 0.0, 10_000_000.0, 2_000_000.0, 10_000.0)
    sqft = st.number_input("Size (sqft)", 0.0, 10000.0, 1200.0, 10.0)
    appr = st.slider("Appreciation % / year", 0.0, 20.0, 8.0)
    rent = st.number_input("Annual Rent (AED)", 0.0, 500_000.0, 120_000.0, 5_000.0)
    years = st.slider("Years rented after handover", 0.0, 10.0, 2.0)
    occ = st.slider("Occupancy %", 50, 100, 90)
    eoi = st.number_input("EOI (AED)", 0.0, 100_000.0, 10_000.0)
    handover = st.slider("Handover (months)", 12, 60, 36)
    sale_month = st.slider("Planned Sale Month", 12, 120, 48)
    sims = st.number_input("Monte Carlo Runs", 1000, 20000, 5000, 1000)

plan = [PaymentMilestone("Booking", 10, 0),
        PaymentMilestone("During Construction", 70, 24),
        PaymentMilestone("Handover", 20, 36)]

p = ProjectInput(
    project_name=name, developer=dev, unit_type="2BR",
    base_price_aed=price, size_sqft=sqft, handover_month=handover,
    sale_month_from_start=sale_month, eoi_aed=eoi, payment_plan=plan,
    appreciation_annual_percent=appr, expected_rent_annual_aed=rent,
    years_rented_post_handover=years, occupancy_percent=occ,
)

# Calculate
engine = OffPlanEngine(p)
df = engine.df()
tot = engine.totals()
met = engine.metrics()

# Display KPIs
st.subheader("📊 Key Performance Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Capital ROI %", f"{met['Capital ROI %']:.2f}%")
c2.metric("Rental ROI %", f"{met['Rental ROI %']:.2f}%")
c3.metric("Total ROI %", f"{met['Total ROI %']:.2f}%")
c4.metric("ROE %", f"{met['ROE %']:.2f}%")

# Charts
st.subheader("💰 Cashflow Timeline")
fig = px.bar(df, x="date", y="cashflow_aed", color="description", title="Cashflows Over Time (AED)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🎯 Monte Carlo ROI Distribution")
with st.spinner("Running simulation..."):
    df_sim = simulate_roi_distribution(p, int(sims))
sim_summary = summarize_simulation(df_sim)
hist = px.histogram(df_sim, x="Total ROI %", nbins=40, title="ROI Distribution (Simulated)")
st.plotly_chart(hist, use_container_width=True)
st.json(sim_summary)

st.caption("© 2025 Oracle Intelligence Labs — Prototype Edition. All results are estimates and not financial advice.")
# === PATCH: extra imports ===
from math import sqrt
from scipy.stats import norm
scipy>=1.11
# === PATCH: market & risk models ===
@dataclass
class MarketRegimes:
    # annual means (pp) and vols (% of mean) for each regime
    appr_mu_pp: Dict[str, float] = None   # {'bear':2, 'base':8, 'bull':14}
    appr_vol_pct: Dict[str, float] = None
    rent_mu_pct: Dict[str, float] = None  # scale on rent (e.g. ±%)
    rent_vol_pct: Dict[str, float] = None
    weights: Dict[str, float] = None      # probabilities summing to 1

@dataclass
class RiskToggles:
    rho_price_rent: float = 0.45           # correlation
    handover_delay_mean_m: float = 2.0     # months
    handover_delay_sd_m: float = 1.0
    vacancy_days_per_year: int = 18
    rent_free_months_on_first_lease: float = 0.0
    refinance_prob: float = 0.15
    refinance_rate_delta_pp: float = -0.5  # drop after refi (pp)
    early_exit_prob: float = 0.10
    early_exit_month_shift: int = -6       # sell 6 months earlier
    svc_inflation_pct_pa: float = 4.0
# === PATCH: offline data providers (synthetic but realistic) ===
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
# === PATCH: regime builder ===
def build_regimes_from_data(dld_mu_pp: float) -> MarketRegimes:
    # tie the base regime to DLD-derived appreciation mean
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
# === PATCH: backtest skeleton ===
def backtest_calibrate(pred_series: np.ndarray, realized_series: np.ndarray) -> Dict[str, float]:
    # very simple scaler to match variance; real life = project-level fit
    pred_std = np.std(pred_series); real_std = np.std(realized_series)
    scale = 1.0 if pred_std==0 else max(0.5, min(2.0, real_std / pred_std))
    return {"vol_scale": float(scale), "pred_std": float(pred_std), "real_std": float(real_std)}
# === PATCH: sidebar — market data & risks ===
st.markdown("---")
st.subheader("Market Data (offline mocks)")
area = st.text_input("Area", "Dubai Creek Harbour")
project_name_input = st.text_input("Project (for comps)", name)
unit_type_ui = st.text_input("Unit Type (for rent comps)", "2BR")
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
# === PATCH: build assumptions from mocks ===
if use_mocks:
    dld = mock_dld_comps(area, project_name_input)
    rents = mock_rera_pf(area, unit_type_ui)
    mort = mock_mortgage_quotes(base_price)
    svc_data = mock_service_charges(project_name_input, size_sqft)

    # override current inputs with mock hints (non-destructive: preview-style)
    project.expected_rent_annual_aed = rents["annual_rent"]
    project.occupancy_percent = rents["occupancy"] * 100
    project.service_charges_aed_per_sqft_year = svc_data["svc_psf"]

    regimes = build_regimes_from_data(dld["appreciation_mu_pp"])
else:
    regimes = build_regimes_from_data(project.appreciation_annual_percent)

# normalize weights from sliders
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
# === PATCH: run advanced simulation ===
with st.spinner("Running correlated Monte-Carlo with regimes & risks ..."):
    sim_df = simulate_roi_distribution_advanced(project, regimes, risks, n_iter=int(sims))

arr = sim_df["Total ROI %"].dropna().to_numpy()
summary = {
    "Mean ROI %": float(np.mean(arr)),
    "Median ROI %": float(np.median(arr)),
    "P50 ROI %": pctl(arr, 50),
    "P90 ROI %": pctl(arr, 10),   # lower is worse; use 10th pct as 'P90 downside'
    "CVaR (worst 10%) %": cvar(arr, 10)
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
with st.expander("Backtest & Calibration (offline demo)"):
    # synthetic realized series (normally you'd upload CSV)
    realized = arr + np.random.normal(0, np.std(arr)*0.2, size=len(arr))
    calib = backtest_calibrate(arr, realized)
    st.write("Calibrated volatility scale:", calib["vol_scale"])
    st.caption("Replace synthetic 'realized' with your historical ROI series per project to tune distributions.")
