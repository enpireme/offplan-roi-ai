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

def simulate_roi_distribution(p: ProjectInput, n_iter: int = 10000, appr_sd: float = 0.1, rent_sd: float = 0.15) -> pd.DataFrame:
    results = []
    for _ in range(n_iter):
        sim_p = ProjectInput(**{**p.__dict__})
        sim_p.appreciation_annual_percent = max(0.0, random.gauss(p.appreciation_annual_percent, p.appreciation_annual_percent*appr_sd))
        sim_p.expected_rent_annual_aed = max(0.0, random.gauss(p.expected_rent_annual_aed, p.expected_rent_annual_aed*rent_sd))
        e = OffPlanEngine(sim_p)
        results.append(e.metrics()["Total ROI %"])
    return pd.DataFrame(results, columns=["Total ROI %"])

def summarize_simulation(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {}
    arr = df["Total ROI %"].dropna()
    return {
        "Mean ROI %": float(np.mean(arr)),
        "Median ROI %": float(np.median(arr)),
        "90% CI Lower": float(np.percentile(arr, 5)),
        "90% CI Upper": float(np.percentile(arr, 95)),
    }

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
