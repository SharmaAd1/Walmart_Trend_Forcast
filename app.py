import json
import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from google.genai.errors import APIError
from prophet import Prophet

from db_connect import DatabaseConnectionError, fetch_walmart_sales

REQUIRED_COLUMNS = {
    "Store",
    "Date",
    "Weekly_Sales",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
}
FACTORS = {
    "Temperature": "Temperature (°F)",
    "Fuel_Price": "Fuel price ($)",
    "CPI": "Consumer price index",
    "Unemployment": "Unemployment rate (%)",
}
COLORS = {
    "primary": "#FF4B4B",
    "blue": "#416AAF",
    "secondary": "#3E4675",
}

st.set_page_config(
    page_title="Walmart Sales Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

dark_mode = st.sidebar.toggle("🌙  Dark mode", value=False, key="dark_mode")
THEME = {
    "ink": "#eef3ff" if dark_mode else "#ffffff",
    "muted": "#a5b2ca" if dark_mode else "#e3e9f5",
    "grid": "#34415a" if dark_mode else "rgba(255, 255, 255, 0.22)",
    "surface": "#1b263b" if dark_mode else COLORS["secondary"],
    "canvas": "#0d1526" if dark_mode else COLORS["blue"],
    "border": "#34415a" if dark_mode else "rgba(255, 255, 255, 0.24)",
    "input_border": "#46536c" if dark_mode else "rgba(255, 255, 255, 0.46)",
    "header": "rgba(13, 21, 38, 0.80)" if dark_mode else "rgba(65, 106, 175, 0.92)",
    "glass": "rgba(27, 38, 59, 0.92)" if dark_mode else "rgba(62, 70, 117, 0.94)",
    "shadow": "rgba(0, 0, 0, 0.35)" if dark_mode else "rgba(40, 70, 140, 0.12)",
    # Toggle colors: the "off" track must stand out against the sidebar surface.
    "toggle_off": "#64748b" if dark_mode else "#8a97ad",
    "toggle_border": "#8da0bf" if dark_mode else "#5f6e87",
}

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: __CANVAS__ !important;
        color: __INK__ !important;
    }
    /* Let the animated Three.js background show through */
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMain"], section.main {
        background: transparent !important;
        position: relative;
        z-index: 1;
    }
    #fx-bg {
        position: fixed; inset: 0; width: 100vw; height: 100vh;
        z-index: 0; pointer-events: none; opacity: .9;
    }
    [data-testid="stHeader"] {
        background: __HEADER__ !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }
    [data-testid="stSidebar"] {
        background: __GLASS__ !important;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-right: 1px solid __BORDER__;
        z-index: 3;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }

    /* ---------- Dark mode toggle: always fully visible ---------- */
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] {
        display: flex !important;
        align-items: center;
        gap: 12px;
        padding: 10px 12px !important;
        margin: 0 0 .6rem 0;
        width: 100%;
        background: __SURFACE__;
        border: 1px solid __BORDER__;
        border-radius: 14px;
        box-shadow: 0 4px 14px __SHADOW__;
        cursor: pointer;
        overflow: visible !important;
    }
    /* Track = the only child that holds no text */
    [data-testid="stSidebar"] label[data-baseweb="checkbox"]
        > :is(span, div):not(:has(p)):not(:has([data-testid="stMarkdownContainer"])) {
        flex: 0 0 44px !important;
        min-width: 44px !important;
        width: 44px !important;
        height: 24px !important;
        border-radius: 999px !important;
        background: __TOGGLE_OFF__ !important;
        border: 1.5px solid __TOGGLE_BORDER__ !important;
        box-shadow: inset 0 1px 3px rgba(0,0,0,.25);
        transition: background .3s ease, border-color .3s ease;
        position: relative;
    }
    [data-testid="stSidebar"] label[data-baseweb="checkbox"]
        > :is(span, div):not(:has(p)):not(:has([data-testid="stMarkdownContainer"])) > div {
        width: 18px !important;
        height: 18px !important;
        background: #ffffff !important;
        border-radius: 50% !important;
        box-shadow: 0 1px 4px rgba(0,0,0,.45);
    }
    [data-testid="stSidebar"] label[data-baseweb="checkbox"]:has(input:checked)
        > :is(span, div):not(:has(p)):not(:has([data-testid="stMarkdownContainer"])) {
        background: linear-gradient(135deg, #3867f4, #12a594) !important;
        border-color: #3867f4 !important;
    }
    /* Label text wrapper: normal width, horizontal text */
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] > :is(span, div):has(p),
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] > :is(span, div):has([data-testid="stMarkdownContainer"]) {
        flex: 1 1 auto !important;
        width: auto !important;
        min-width: 0 !important;
        height: auto !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 0 !important;
        white-space: nowrap;
    }
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] > :is(span, div):has(p) > div {
        width: auto !important; height: auto !important;
        background: transparent !important; box-shadow: none !important; border-radius: 0 !important;
    }
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label[data-baseweb="checkbox"] p {
        color: __INK__ !important;
        font-weight: 600;
        margin: 0;
    }

    /* ---------- Inputs ---------- */
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="select"] > div,
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="input"] > div,
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-testid="stDateInput"] button,
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-testid="stDateInput"] input {
        background-color: __SURFACE__ !important;
        border-color: __INPUT_BORDER__ !important;
        color: __INK__ !important;
        border-radius: 12px !important;
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="select"] > div:hover,
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="select"] > div:focus-within {
        border-color: #3867f4 !important;
        box-shadow: 0 0 0 3px rgba(56, 103, 244, .18);
    }
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"],
    [role="listbox"],
    [data-testid="stMainMenuPopover"] {
        background-color: __SURFACE__ !important;
        border-color: __BORDER__ !important;
        color: __INK__ !important;
    }
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="listbox"] *,
    [data-testid="stMainMenuPopover"] * {
        color: __INK__ !important;
    }
    [data-baseweb="calendar"],
    [data-baseweb="calendar"] table,
    [data-baseweb="calendar"] [role="grid"],
    [data-baseweb="calendar"] [role="row"],
    [data-baseweb="calendar"] [role="gridcell"] {
        background-color: __SURFACE__ !important;
        color: __INK__ !important;
    }
    [data-baseweb="calendar"] button {
        background-color: transparent !important;
        color: __INK__ !important;
    }
    [data-baseweb="calendar"] button:hover {
        background-color: __BORDER__ !important;
    }
    [data-baseweb="calendar"] [aria-selected="true"],
    [data-baseweb="calendar"] [aria-selected="true"] button {
        background: linear-gradient(135deg, #3867f4, #12a594) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stChatInput"] textarea {
        background: __SURFACE__ !important;
        border-color: __INPUT_BORDER__ !important;
        color: __INK__ !important;
        -webkit-text-fill-color: __INK__ !important;
    }
    [data-testid="stSidebar"] [data-testid="stChatInput"] button {
        background: linear-gradient(135deg, #3867f4, #12a594) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stChatMessage"] {
        background: __GLASS__ !important;
        border: 0 !important;
        border-radius: 12px;
        padding: 8px 10px;
    }
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="select"] input,
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="input"] input {
        color: __INK__ !important;
        -webkit-text-fill-color: __INK__ !important;
    }
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-baseweb="select"] svg {
        fill: __MUTED__ !important;
    }
    :is([data-testid="stAppViewContainer"], [data-testid="stSidebar"]) [data-testid="stSlider"] [role="slider"] {
        background-color: #3867f4 !important;
        border-color: __SURFACE__ !important;
        box-shadow: 0 0 0 4px rgba(56, 103, 244, .22);
    }
    [data-testid="stAppViewContainer"] *,
    [data-testid="stSidebar"] * {
        color: __INK__;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] {
        color: __MUTED__ !important;
    }
    .block-container {
        max-width: 1480px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* ---------- Headings ---------- */
    h1, h2, h3,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: __INK__ !important;
        letter-spacing: -0.025em;
    }
    .block-container h3 {
        position: relative;
        padding-left: 14px;
    }
    .block-container h3::before {
        content: "";
        position: absolute; left: 0; top: .28em; bottom: .28em; width: 4px;
        border-radius: 4px;
        background: linear-gradient(180deg, #3867f4, #12a594);
    }

    /* ---------- Cards ---------- */
    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        background: __GLASS__ !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid __BORDER__;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px __SHADOW__;
        will-change: transform;
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute; left: 0; top: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #3867f4, #12a594);
    }
    [data-testid="stMetric"]::after {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(260px circle at var(--mx, 85%) var(--my, 0%),
                    rgba(56,103,244,.30), rgba(18,165,148,.10) 45%, transparent 70%);
        opacity: .45;
        transition: opacity .3s ease;
        pointer-events: none;
    }
    [data-testid="stMetric"]:hover {
        border-color: #3867f4;
        box-shadow: 0 0 0 1px #3867f4, 0 18px 40px rgba(56,103,244,.35), 0 0 28px rgba(18,165,148,.25);
    }
    [data-testid="stMetric"]:hover::after { opacity: 1; }
    [data-testid="stMetric"]:hover::before { height: 5px; }
    [data-testid="stMetric"]::before { transition: height .25s ease; }
    [data-testid="stMetric"] > * { position: relative; z-index: 1; }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: __MUTED__ !important;
    }
    [data-testid="stMetricLabel"] div {
        color: __MUTED__ !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.25;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div {
        color: __INK__ !important;
    }
    [data-testid="stMetricValue"] div {
        font-size: 1.65rem !important;
        font-weight: 700;
        line-height: 1.2 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stPlotlyChart"] {
        background: __GLASS__;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid __BORDER__;
        border-radius: 18px;
        padding: 10px;
        overflow: hidden;
        box-shadow: 0 8px 24px __SHADOW__;
    }
    .dashboard-eyebrow {
        color: #3867f4 !important;
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        line-height: 1.4;
        margin: 0.25rem 0 0.65rem;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(56, 103, 244, .12);
        border: 1px solid rgba(56, 103, 244, .3);
    }
    .block-container h1 {
        line-height: 1.15 !important;
        margin-top: 0 !important;
        margin-bottom: 0.9rem !important;
    }
    [data-testid="stCaptionContainer"] {
        color: __MUTED__ !important;
    }
    [data-testid="stDataFrame"], [data-testid="stExpander"] {
        border-color: __BORDER__ !important;
    }
    [data-testid="stExpander"] {
        background: __GLASS__;
        border-radius: 16px;
        backdrop-filter: blur(14px);
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #3867f4, #12a594) !important;
        border: 0 !important;
        border-radius: 12px !important;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .stDownloadButton button, .stDownloadButton button * { color: #ffffff !important; }
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(56, 103, 244, .4);
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: __INPUT_BORDER__; border-radius: 10px; }
    @media (prefers-reduced-motion: no-preference) {
        [data-testid="stPlotlyChart"] {
            transition: border-color 240ms ease, box-shadow 240ms ease, transform 240ms ease;
        }
        [data-testid="stPlotlyChart"]:hover {
            transform: translateY(-3px);
            border-color: rgba(56, 103, 244, .55);
            box-shadow: 0 16px 36px __SHADOW__;
        }
    }
    </style>
    """
    .replace("__CANVAS__", THEME["canvas"])
    .replace("__INK__", THEME["ink"])
    .replace("__MUTED__", THEME["muted"])
    .replace("__HEADER__", THEME["header"])
    .replace("__SURFACE__", THEME["surface"])
    .replace("__GLASS__", THEME["glass"])
    .replace("__SHADOW__", THEME["shadow"])
    .replace("__BORDER__", THEME["border"])
    .replace("__INPUT_BORDER__", THEME["input_border"])
    .replace("__TOGGLE_OFF__", THEME["toggle_off"])
    .replace("__TOGGLE_BORDER__", THEME["toggle_border"])
    .replace("#3867f4", COLORS["primary"])
    .replace("#12a594", COLORS["blue"])
    .replace("rgba(56,103,244", "rgba(255,75,75")
    .replace("rgba(18,165,148", "rgba(65,106,175")
    .replace("rgba(56, 103, 244", "rgba(255, 75, 75")
    .replace("rgba(18, 165, 148", "rgba(65, 106, 175"),
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading Walmart sales data...")
def load_sales_data() -> pd.DataFrame:
    data = fetch_walmart_sales()
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"The sales data is missing required columns: {missing}")

    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
    if data["Date"].isna().any():
        raise ValueError("The sales data contains dates that could not be parsed.")

    numeric_columns = sorted(REQUIRED_COLUMNS - {"Date"})
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[numeric_columns].isna().any().any():
        raise ValueError("The sales data contains missing or non-numeric values.")

    data["Store"] = data["Store"].astype(int)
    data["Holiday_Flag"] = data["Holiday_Flag"].astype(int)
    return data.sort_values(["Date", "Store"]).reset_index(drop=True)


@st.cache_data(show_spinner="Building the store forecast...")
def build_store_forecast(
    store_history: pd.DataFrame, periods: int
) -> pd.DataFrame:
    training_data = store_history[["Date", "Weekly_Sales"]].rename(
        columns={"Date": "ds", "Weekly_Sales": "y"}
    )
    model = Prophet(
        interval_width=0.8,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(training_data)
    future_dates = model.make_future_dataframe(periods=periods, freq="W-FRI")
    return model.predict(future_dates)


def format_currency(value: float) -> str:
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.2f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.2f}M"
    return f"${value:,.0f}"


def query_walmart_sales(
    sales_data: pd.DataFrame,
    operation: str,
    store: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, object]:
    if operation not in {"summary", "top_stores", "weekly_totals", "records"}:
        raise ValueError(f"Unsupported sales query operation: {operation}")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")
    if offset < 0:
        raise ValueError("offset must be zero or greater.")

    filtered = sales_data
    if store is not None:
        filtered = filtered.loc[filtered["Store"] == store]
    for boundary_name, boundary in (
        ("start_date", start_date),
        ("end_date", end_date),
    ):
        if boundary is not None:
            try:
                date_value = pd.Timestamp(boundary)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{boundary_name} must be a valid date.") from error
            if boundary_name == "start_date":
                filtered = filtered.loc[filtered["Date"] >= date_value]
            else:
                filtered = filtered.loc[filtered["Date"] <= date_value]

    if operation == "summary":
        if filtered.empty:
            return {"row_count": 0, "message": "No matching sales records."}
        sales_correlations = {}
        for column in ("Temperature", "Fuel_Price", "CPI", "Unemployment"):
            correlation = filtered["Weekly_Sales"].corr(filtered[column])
            sales_correlations[column] = (
                float(correlation) if pd.notna(correlation) else None
            )
        holiday_sales = filtered.loc[
            filtered["Holiday_Flag"] == 1, "Weekly_Sales"
        ]
        non_holiday_sales = filtered.loc[
            filtered["Holiday_Flag"] == 0, "Weekly_Sales"
        ]
        return {
            "row_count": len(filtered),
            "date_range": [
                filtered["Date"].min().strftime("%Y-%m-%d"),
                filtered["Date"].max().strftime("%Y-%m-%d"),
            ],
            "stores": sorted(int(value) for value in filtered["Store"].unique()),
            "total_weekly_sales": float(filtered["Weekly_Sales"].sum()),
            "average_weekly_sales": float(filtered["Weekly_Sales"].mean()),
            "holiday_rows": int(filtered["Holiday_Flag"].sum()),
            "non_holiday_rows": int((filtered["Holiday_Flag"] == 0).sum()),
            "average_sales_by_week_type": {
                "holiday": (
                    float(holiday_sales.mean()) if not holiday_sales.empty else None
                ),
                "non_holiday": (
                    float(non_holiday_sales.mean())
                    if not non_holiday_sales.empty
                    else None
                ),
            },
            "average_market_factors": {
                column: float(filtered[column].mean())
                for column in ("Temperature", "Fuel_Price", "CPI", "Unemployment")
            },
            "sales_correlations_with_market_factors": sales_correlations,
        }

    if operation == "top_stores":
        grouped = (
            filtered.groupby("Store", as_index=False)["Weekly_Sales"]
            .sum()
            .sort_values("Weekly_Sales", ascending=False)
        )
        return {
            "total_stores": len(grouped),
            "stores": json.loads(
                grouped.iloc[offset : offset + limit].to_json(orient="records")
            ),
        }

    if operation == "weekly_totals":
        grouped = (
            filtered.groupby("Date", as_index=False)["Weekly_Sales"]
            .sum()
            .sort_values("Date")
        )
        return {
            "total_weeks": len(grouped),
            "weeks": json.loads(
                grouped.iloc[offset : offset + limit].to_json(
                    orient="records", date_format="iso"
                )
            ),
        }

    rows = filtered.iloc[offset : offset + limit]
    return {
        "total_matching_rows": len(filtered),
        "offset": offset,
        "limit": limit,
        "records": json.loads(rows.to_json(orient="records", date_format="iso")),
    }


WALMART_SALES_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="query_walmart_sales",
            description=(
                "Read the Walmart sales table. Use this for factual questions about "
                "sales records, stores, dates, holidays, or market factors."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "operation": types.Schema(
                        type=types.Type.STRING,
                        enum=["summary", "top_stores", "weekly_totals", "records"],
                        description=(
                            "summary for metrics, top_stores for ranked store totals, "
                            "weekly_totals for totals by date, records for raw rows."
                        ),
                    ),
                    "store": types.Schema(
                        type=types.Type.INTEGER,
                        description="Optional store number filter.",
                    ),
                    "start_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional inclusive date filter in YYYY-MM-DD format.",
                    ),
                    "end_date": types.Schema(
                        type=types.Type.STRING,
                        description="Optional inclusive date filter in YYYY-MM-DD format.",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum rows to return, from 1 to 100.",
                    ),
                    "offset": types.Schema(
                        type=types.Type.INTEGER,
                        description="Number of matching rows to skip for pagination.",
                    ),
                },
                required=["operation"],
            ),
        )
    ]
)


def style_figure(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=height,
        margin=dict(l=12, r=16, t=28, b=12),
        font=dict(family="Inter, sans-serif", color=THEME["ink"], size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=THEME["ink"], font_color=THEME["surface"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        transition=dict(duration=600, easing="cubic-in-out"),
    )
    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=THEME["muted"]),
        title_font=dict(color=THEME["ink"]),
    )
    figure.update_yaxes(
        gridcolor=THEME["grid"],
        zeroline=False,
        tickfont=dict(color=THEME["muted"]),
        title_font=dict(color=THEME["ink"]),
    )
    return figure


try:
    sales = load_sales_data()
except DatabaseConnectionError as error:
    st.error(str(error))
    st.stop()

minimum_date = sales["Date"].min().date()
maximum_date = sales["Date"].max().date()
all_stores = sorted(sales["Store"].unique().tolist())

st.sidebar.markdown("## 📊 Sales intelligence")
st.sidebar.caption("Explore Walmart weekly sales by store, time, and market factor.")
date_selection = st.sidebar.date_input(
    "Date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)
selected_stores = st.sidebar.multiselect(
    "Stores",
    options=all_stores,
    default=[],
    placeholder="All stores",
    help="Leave empty to include all stores.",
    format_func=lambda store: f"Store {store:02d}",
)
forecast_horizon = st.sidebar.slider(
    "Forecast horizon (weeks)", min_value=4, max_value=52, value=12, step=4
)
selected_factor = st.sidebar.selectbox(
    "Market factor", options=list(FACTORS), format_func=FACTORS.get
)

st.sidebar.markdown("### ♊ Gemini assistant")
with st.sidebar.container(height=260, border=False):
    for message in st.session_state.get("gemini_messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if "gemini_messages" not in st.session_state:
    st.session_state.gemini_messages = []

if prompt := st.sidebar.chat_input("Ask Gemini anything...", key="gemini_chat_input"):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = None
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.sidebar.error(
            "Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets "
            "or the environment."
        )
    else:
        contents = [
            types.Content(
                role="user" if message["role"] == "user" else "model",
                parts=[types.Part(text=message["content"])],
            )
            for message in st.session_state.gemini_messages
        ]
        contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
        generation_config = types.GenerateContentConfig(
            system_instruction=(
                "You are the Walmart sales dashboard assistant. The full loaded "
                "walmart_sales table is available through the query_walmart_sales "
                "tool. Use that tool for factual claims about the data, and never "
                "invent results. The tool is read-only and does not expose database "
                "credentials or source code. Clearly state when results are paginated."
            ),
            tools=[WALMART_SALES_TOOL],
        )

        try:
            with st.sidebar:
                with st.spinner("Gemini is thinking..."):
                    with genai.Client(api_key=api_key) as client:
                        for _ in range(5):
                            response = client.models.generate_content(
                                model="gemini-3.1-flash-lite",
                                contents=contents,
                                config=generation_config,
                            )
                            function_calls = response.function_calls or []
                            if not function_calls:
                                break

                            function_responses = []
                            for function_call in function_calls:
                                if function_call.name != "query_walmart_sales":
                                    result = {"error": "Unsupported database operation."}
                                else:
                                    try:
                                        result = query_walmart_sales(
                                            sales,
                                            **(function_call.args or {}),
                                        )
                                    except (TypeError, ValueError) as error:
                                        result = {"error": str(error)}
                                function_responses.append(
                                    types.Part.from_function_response(
                                        name=function_call.name,
                                        response=result,
                                    )
                                )
                            contents.extend(
                                [
                                    response.candidates[0].content,
                                    types.Content(
                                        role="tool",
                                        parts=function_responses,
                                    ),
                                ]
                            )
                        else:
                            st.sidebar.error(
                                "Gemini could not complete the database lookup. "
                                "Please try a narrower question."
                            )
                            response = None

            if response is not None:
                answer = response.text
                if not answer:
                    st.sidebar.error(
                        "Gemini returned an empty response. Please try again."
                    )
                else:
                    st.session_state.gemini_messages.extend(
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer},
                        ]
                    )
                    st.rerun()
        except APIError as error:
            st.sidebar.error(f"Gemini request failed: {error}")

if len(date_selection) != 2:
    st.info("Select both a start and end date to display the dashboard.")
    st.stop()
if not selected_stores:
    stores_in_view = all_stores
else:
    stores_in_view = selected_stores

start_date, end_date = date_selection
filtered_sales = sales.loc[
    sales["Date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    & sales["Store"].isin(stores_in_view)
].copy()
if filtered_sales.empty:
    st.info("No sales records match the selected filters.")
    st.stop()

# ----------------------------------------------------------------------------
# Hero: Three.js scene + GSAP timeline + anime.js micro-animations
# ----------------------------------------------------------------------------
HERO_HTML = """
<div class="hero" id="hero">
  <canvas id="hero-scene" aria-hidden="true"></canvas>
  <div class="glow g1"></div><div class="glow g2"></div>
  <div class="hero-copy">
    <div class="hero-kicker"><span class="pulse"></span> Live weekly retail analytics</div>
    <h1 id="hero-title">Walmart Sales Intelligence</h1>
    <p>Track sales performance, compare stores, see how market conditions move demand, and forecast the weeks ahead.</p>
    <div class="chips">
      <span class="chip">Store rankings</span>
      <span class="chip">Holiday impact</span>
      <span class="chip">Market factors</span>
      <span class="chip">Prophet forecasts</span>
    </div>
  </div>
  <div class="hero-orbit" aria-hidden="true"><span></span><span></span><span></span><i></i></div>
</div>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: transparent; font-family: Inter, system-ui, sans-serif; }
  .hero {
    position: relative; overflow: hidden; min-height: 226px; padding: 30px 36px;
    border-radius: 24px; color: __INK__;
    background: linear-gradient(120deg, __SURFACE__ 0%, __SURFACE2__ 100%);
    border: 1px solid __BORDER__;
    box-shadow: 0 14px 40px __SHADOW__;
  }
  #hero-scene { position: absolute; inset: 0; width: 100%; height: 100%; opacity: .95; }
  .glow { position: absolute; border-radius: 50%; filter: blur(60px); opacity: .5; pointer-events: none; }
  .g1 { width: 260px; height: 260px; left: -60px; top: -90px; background: rgba(56,103,244,.45); }
  .g2 { width: 240px; height: 240px; right: 8%; bottom: -120px; background: rgba(18,165,148,.4); }
  .hero-copy { position: relative; z-index: 2; max-width: 720px; }
  .hero-kicker { color: #3867f4; font-size: 12px; font-weight: 700; letter-spacing: .06em; display: flex; align-items: center; }
  .hero-kicker .pulse {
    display: inline-block; width: 9px; height: 9px; margin-right: 9px;
    border-radius: 50%; background: #12a594; box-shadow: 0 0 14px #12a594;
  }
  .hero h1 {
    margin: 14px 0 10px; font-size: clamp(28px, 3.6vw, 46px); line-height: 1.1; letter-spacing: -.04em;
    background: linear-gradient(90deg, __INK__ 0%, #3867f4 55%, #12a594 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
  }
  .hero h1 .ch { display: inline-block; }
  .hero p { margin: 0; color: __MUTED__; font-size: 14.5px; line-height: 1.6; max-width: 600px; }
  .chips { margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; }
  .chip {
    padding: 6px 13px; font-size: 12px; font-weight: 600; border-radius: 999px; color: __INK__;
    background: __CHIP__; border: 1px solid __BORDER__; backdrop-filter: blur(6px);
  }
  .hero-orbit { position: absolute; width: 170px; height: 170px; right: 6%; top: 28px; z-index: 1; }
  .hero-orbit span { position: absolute; inset: 0; border: 1px solid rgba(56,103,244,.35); border-radius: 50%; }
  .hero-orbit span:nth-child(2) { inset: 20px; border-color: rgba(18,165,148,.5); }
  .hero-orbit span:nth-child(3) { inset: 42px; border-color: rgba(56,103,244,.6); }
  .hero-orbit i { position: absolute; left: 50%; top: 50%; width: 12px; height: 12px; margin: -6px;
    border-radius: 50%; background: linear-gradient(135deg,#3867f4,#12a594); box-shadow: 0 0 18px #3867f4; }
  @media (max-width: 700px) { .hero { padding: 24px; } .hero-orbit { right: -60px; opacity: .45; } }
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"></script>
<script>
(() => {
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dark = __DARK__;
  const hero = document.getElementById("hero");
  const canvas = document.getElementById("hero-scene");

  // Split the title into characters for the GSAP entrance.
  const title = document.getElementById("hero-title");
  title.innerHTML = [...title.textContent].map(c => c === " " ? " " : '<span class="ch">' + c + "</span>").join("");

  /* ---------------- Three.js ---------------- */
  if (!reduced && window.THREE && canvas) {
    try {
      const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
      camera.position.z = 6;

      const group = new THREE.Group();
      scene.add(group);

      // Particle field
      const pGeo = new THREE.BufferGeometry();
      const pos = [];
      for (let i = 0; i < 260; i++) pos.push((Math.random() - .5) * 14, (Math.random() - .5) * 5, (Math.random() - .5) * 5);
      pGeo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      const points = new THREE.Points(pGeo, new THREE.PointsMaterial({
        color: 0x3867f4, size: .035, transparent: true, opacity: .85 }));
      group.add(points);

      // Wireframe objects on the right side
      const knot = new THREE.Mesh(
        new THREE.TorusKnotGeometry(.9, .26, 120, 16),
        new THREE.MeshBasicMaterial({ color: 0x12a594, wireframe: true, transparent: true, opacity: dark ? .45 : .35 }));
      knot.position.set(3.4, 0, -1);
      group.add(knot);

      const ico = new THREE.Mesh(
        new THREE.IcosahedronGeometry(.7, 1),
        new THREE.MeshBasicMaterial({ color: 0x3867f4, wireframe: true, transparent: true, opacity: .5 }));
      ico.position.set(-3.6, -.6, -1.5);
      group.add(ico);

      // Rising "sales bars" for a retail-data feel
      const bars = [];
      for (let i = 0; i < 14; i++) {
        const h = 0.4 + Math.random() * 1.4;
        const bar = new THREE.Mesh(
          new THREE.BoxGeometry(.14, h, .14),
          new THREE.MeshBasicMaterial({ color: i % 2 ? 0x3867f4 : 0x12a594, transparent: true, opacity: .35 }));
        bar.position.set(-1.2 + i * .28, -1.9 + h / 2, -2);
        bar.userData = { base: h, phase: Math.random() * Math.PI * 2 };
        group.add(bar);
        bars.push(bar);
      }

      const resize = () => {
        const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
      };
      new ResizeObserver(resize).observe(canvas);
      resize();

      const mouse = { x: 0, y: 0 };
      hero.addEventListener("mousemove", e => {
        const r = hero.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width - .5) * 2;
        mouse.y = ((e.clientY - r.top) / r.height - .5) * 2;
      });

      const clock = new THREE.Clock();
      const tick = () => {
        const t = clock.getElapsedTime();
        points.rotation.y = t * .03;
        knot.rotation.x = t * .3; knot.rotation.y = t * .4;
        ico.rotation.x = -t * .25; ico.rotation.y = t * .35;
        bars.forEach(b => {
          const s = .75 + Math.sin(t * 1.2 + b.userData.phase) * .25;
          b.scale.y = s;
          b.position.y = -1.9 + (b.userData.base * s) / 2;
        });
        group.rotation.y += (mouse.x * .22 - group.rotation.y) * .04;
        group.rotation.x += (mouse.y * .12 - group.rotation.x) * .04;
        renderer.render(scene, camera);
        requestAnimationFrame(tick);
      };
      tick();
    } catch (err) { console.warn("Three.js hero disabled:", err); }
  }

  /* ---------------- GSAP: entrance timeline ---------------- */
  if (!reduced && window.gsap) {
    const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
    tl.from(".hero", { opacity: 0, y: 24, scale: .985, duration: .8 })
      .from(".hero-kicker", { x: -20, opacity: 0, duration: .6 }, "-=.4")
      .from(".hero h1 .ch", { yPercent: 110, rotateX: -70, opacity: 0, duration: .7, stagger: .025 }, "-=.3")
      .from(".hero p", { y: 16, opacity: 0, duration: .6 }, "-=.35")
      .from(".chip", { y: 14, opacity: 0, scale: .9, duration: .5, stagger: .08, ease: "back.out(2)" }, "-=.3");
    gsap.to(".hero-orbit", { rotation: 360, duration: 30, repeat: -1, ease: "none" });
    gsap.to(".hero h1", { backgroundPosition: "200% 0", duration: 6, repeat: -1, yoyo: true, ease: "sine.inOut" });
    gsap.to(".glow.g1", { x: 40, y: 24, duration: 7, repeat: -1, yoyo: true, ease: "sine.inOut" });
    gsap.to(".glow.g2", { x: -50, y: -20, duration: 8, repeat: -1, yoyo: true, ease: "sine.inOut" });
  }

  /* ---------------- anime.js: pulse + orbit rings + chip hover ---------------- */
  if (!reduced && window.anime) {
    anime({ targets: ".hero-kicker .pulse", scale: [1, 1.5], opacity: [.6, 1],
            duration: 1000, direction: "alternate", loop: true, easing: "easeInOutSine" });
    anime({ targets: ".hero-orbit span", rotate: ["0deg", "360deg"],
            delay: anime.stagger(250), duration: 14000, direction: "alternate", loop: true, easing: "easeInOutSine" });
    anime({ targets: ".hero-orbit i", scale: [1, 1.8], duration: 1600,
            direction: "alternate", loop: true, easing: "easeInOutQuad" });
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("mouseenter", () => anime({ targets: chip, translateY: -4, duration: 250, easing: "easeOutQuad" }));
      chip.addEventListener("mouseleave", () => anime({ targets: chip, translateY: 0, duration: 350, easing: "easeOutElastic(1, .6)" }));
    });
  }
})();
</script>
"""

st.markdown('<div class="dashboard-eyebrow">Retail performance overview</div>', unsafe_allow_html=True)
components.html(
    HERO_HTML.replace("__INK__", THEME["ink"])
    .replace("__MUTED__", THEME["muted"])
    .replace("__SURFACE__", THEME["surface"])
    .replace("__SURFACE2__", "#16223a" if dark_mode else "#557bb7")
    .replace("__BORDER__", THEME["border"])
    .replace("__SHADOW__", THEME["shadow"])
    .replace("__CHIP__", "rgba(255,255,255,.06)" if dark_mode else "rgba(255,255,255,.7)")
    .replace("__DARK__", "true" if dark_mode else "false")
    .replace("#3867f4", COLORS["primary"])
    .replace("#12a594", COLORS["blue"])
    .replace("rgba(56,103,244", "rgba(255,75,75")
    .replace("rgba(18,165,148", "rgba(65,106,175")
    .replace("rgba(56, 103, 244", "rgba(255, 75, 75")
    .replace("rgba(18, 165, 148", "rgba(65, 106, 175"),
    height=246,
    scrolling=False,
)

total_sales = filtered_sales["Weekly_Sales"].sum()
average_weekly_sales = filtered_sales["Weekly_Sales"].mean()
best_store = (
    filtered_sales.groupby("Store")["Weekly_Sales"].sum().idxmax()
)
weeks_covered = filtered_sales["Date"].nunique()

metric_columns = st.columns(4)
metric_columns[0].metric("Total sales", format_currency(total_sales))
metric_columns[1].metric("Average store-week", format_currency(average_weekly_sales))
metric_columns[2].metric("Stores in view", f"{filtered_sales['Store'].nunique():,}")
metric_columns[3].metric(
    "Top store", f"Store {best_store:02d}", help=f"Across {weeks_covered} weeks"
)

st.markdown("### Sales over time")
st.caption("Combined weekly sales for the selected stores, with a 4-week moving average.")
weekly_sales = (
    filtered_sales.groupby("Date", as_index=False)[["Weekly_Sales"]]
    .sum()
    .sort_values("Date")
)
weekly_sales["4-week average"] = weekly_sales["Weekly_Sales"].rolling(
    window=4, min_periods=1
).mean()
trend_figure = go.Figure()
trend_figure.add_trace(
    go.Scatter(
        x=weekly_sales["Date"],
        y=weekly_sales["Weekly_Sales"],
        mode="lines",
        name="Weekly sales",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255, 75, 75, 0.08)",
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Weekly sales</extra>",
    )
)
trend_figure.add_trace(
    go.Scatter(
        x=weekly_sales["Date"],
        y=weekly_sales["4-week average"],
        mode="lines",
        name="4-week average",
        line=dict(color=COLORS["primary"], width=3),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>4-week average</extra>",
    )
)
trend_figure.update_layout(hovermode="x unified", yaxis_title="Sales ($)", xaxis_title=None)
st.plotly_chart(style_figure(trend_figure, 400), use_container_width=True, theme=None)

left_column, right_column = st.columns(2)
with left_column:
    st.markdown("### Store performance")
    st.caption("Top stores by total sales in the current view.")
    store_performance = (
        filtered_sales.groupby("Store", as_index=False)[["Weekly_Sales"]]
        .sum()
        .nlargest(12, "Weekly_Sales")
        .sort_values("Weekly_Sales")
    )
    store_figure = px.bar(
        store_performance,
        x="Weekly_Sales",
        y=store_performance["Store"].map(lambda store: f"Store {store:02d}"),
        orientation="h",
        text="Weekly_Sales",
        color_discrete_sequence=[COLORS["blue"]],
        labels={"Weekly_Sales": "Total sales ($)", "y": "Store"},
    )
    store_figure.update_traces(
        texttemplate="$%{x:.4s}",
        textposition="outside",
        cliponaxis=False,
        marker_cornerradius=6,
        hovertemplate="%{y}<br>$%{x:,.0f}<extra></extra>",
    )
    store_figure.update_layout(showlegend=False, xaxis_title="Total sales ($)", yaxis_title=None)
    store_figure.update_xaxes(range=[0, store_performance["Weekly_Sales"].max() * 1.22])
    store_figure.update_layout(margin=dict(l=12, r=28, t=28, b=12))
    st.plotly_chart(style_figure(store_figure, 400), use_container_width=True, theme=None)

with right_column:
    st.markdown("### Holiday vs. regular weeks")
    st.caption("Average weekly sales per store, with the selected date and store filters.")
    holiday_data = filtered_sales.assign(
        Week_Type=filtered_sales["Holiday_Flag"].map(
            {0: "Regular week", 1: "Holiday week"}
        )
    )
    holiday_summary = (
        holiday_data.groupby("Week_Type", as_index=False)[["Weekly_Sales"]]
        .mean()
        .rename(columns={"Weekly_Sales": "Average_Sales"})
    )
    holiday_figure = px.bar(
        holiday_summary,
        x="Week_Type",
        y="Average_Sales",
        color="Week_Type",
        color_discrete_map={
            "Regular week": COLORS["blue"],
            "Holiday week": COLORS["primary"],
        },
        text="Average_Sales",
        labels={"Week_Type": "Week type", "Average_Sales": "Average weekly sales ($)"},
    )
    holiday_figure.update_traces(
        texttemplate="$%{y:,.0f}",
        textposition="outside",
        marker_cornerradius=8,
        hovertemplate="%{x}<br>$%{y:,.0f} per store-week<extra></extra>",
    )
    holiday_figure.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Average sales ($)")
    holiday_figure.update_yaxes(range=[0, holiday_summary["Average_Sales"].max() * 1.18])
    st.plotly_chart(style_figure(holiday_figure, 400), use_container_width=True, theme=None)

st.markdown("### Sales and market conditions")
st.caption("Each point represents one store-week. Hover for store, date, and sales details.")
factor_data = filtered_sales.assign(
    Week_Type=filtered_sales["Holiday_Flag"].map(
        {0: "Regular week", 1: "Holiday week"}
    )
)
factor_figure = px.scatter(
    factor_data,
    x=selected_factor,
    y="Weekly_Sales",
    color="Week_Type",
    color_discrete_map={
        "Regular week": COLORS["blue"],
        "Holiday week": COLORS["primary"],
    },
    hover_data={"Store": True, "Date": "|%b %d, %Y", "Weekly_Sales": ":$,.0f"},
    labels={
        selected_factor: FACTORS[selected_factor],
        "Weekly_Sales": "Weekly sales ($)",
        "Week_Type": "Week type",
    },
    opacity=0.68,
)
factor_figure.update_traces(marker=dict(size=7))
factor_figure.update_layout(
    legend_title_text=None,
    xaxis_title=FACTORS[selected_factor],
    yaxis_title="Weekly sales ($)",
)
st.plotly_chart(style_figure(factor_figure, 420), use_container_width=True, theme=None)

st.markdown("### Store-level sales forecast")
st.caption(
    "Prophet forecast trained on the complete history for the selected store. "
    "The date filter does not truncate forecast training data."
)
forecast_store = st.selectbox(
    "Store to forecast",
    options=stores_in_view,
    format_func=lambda store: f"Store {store:02d}",
    key="forecast_store",
)
store_history = sales.loc[
    sales["Store"] == forecast_store, ["Date", "Weekly_Sales"]
].copy()
forecast = build_store_forecast(store_history, forecast_horizon)
last_observation = store_history.sort_values("Date").iloc[-1]
future_forecast = forecast.loc[forecast["ds"] > last_observation["Date"]].copy()

forecast_figure = go.Figure()
forecast_figure.add_trace(
    go.Scatter(
        x=store_history["Date"],
        y=store_history["Weekly_Sales"],
        mode="lines",
        name="Actual sales",
        line=dict(color=THEME["ink"], width=2),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Actual</extra>",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 75, 75, 0.14)",
        name="80% confidence interval",
        hoverinfo="skip",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=[last_observation["Date"], *future_forecast["ds"].tolist()],
        y=[last_observation["Weekly_Sales"], *future_forecast["yhat"].tolist()],
        mode="lines",
        name="Forecast",
        line=dict(color=COLORS["blue"], width=3, dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:,.0f}<extra>Forecast</extra>",
    )
)
forecast_figure.update_layout(
    hovermode="x unified",
    xaxis_title=None,
    yaxis_title="Weekly sales ($)",
)
st.plotly_chart(style_figure(forecast_figure, 420), use_container_width=True, theme=None)

forecast_metrics = st.columns(3)
forecast_metrics[0].metric(
    "Last recorded week", format_currency(float(last_observation["Weekly_Sales"]))
)
forecast_metrics[1].metric(
    f"Average forecast · {forecast_horizon} weeks",
    format_currency(float(future_forecast["yhat"].mean())),
)
forecast_metrics[2].metric(
    "Forecast change",
    f"{(future_forecast['yhat'].iloc[-1] / last_observation['Weekly_Sales'] - 1) * 100:+.1f}%",
    help="Change from the latest recorded week to the final forecast week.",
)

with st.expander("Explore filtered records"):
    st.dataframe(
        filtered_sales,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Weekly_Sales": st.column_config.NumberColumn("Weekly sales", format="$%.2f"),
            "Date": st.column_config.DateColumn("Date", format="MMM DD, YYYY"),
        },
    )
    st.download_button(
        "Download filtered records",
        data=filtered_sales.to_csv(index=False).encode("utf-8"),
        file_name="walmart_sales_filtered.csv",
        mime="text/csv",
    )

# ----------------------------------------------------------------------------
# Page-wide motion controller (invisible iframe that animates the parent page):
#  - Three.js: fixed animated particle background behind the whole app
#  - GSAP: scroll-triggered reveal + 3D tilt on cards, sidebar entrance
#  - anime.js: metric count-up numbers
# ----------------------------------------------------------------------------
FX_JS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.5/gsap.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.2/anime.min.js"></script>
<script>
(() => {
  let P, D;
  try { P = window.parent; D = P.document; D.body; } catch (e) { return; }
  const reduced = P.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const dark = __DARK__;
  if (reduced) return;

  // Clean up anything left by a previous run (theme switch / rerun)
  if (P.__fxCleanup) { try { P.__fxCleanup(); } catch (e) {} }
  const cleanups = [];
  P.__fxCleanup = () => cleanups.forEach(fn => { try { fn(); } catch (e) {} });
  window.addEventListener("pagehide", () => P.__fxCleanup && P.__fxCleanup());

  /* ---------- 1. Three.js ambient background ---------- */
  try {
    const canvas = D.createElement("canvas");
    canvas.id = "fx-bg";
    D.body.prepend(canvas);
    cleanups.push(() => canvas.remove());

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(P.devicePixelRatio || 1, 1.5));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 200);
    camera.position.z = 14;

    const N = 420, arr = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      arr[i*3] = (Math.random() - .5) * 40;
      arr[i*3+1] = (Math.random() - .5) * 24;
      arr[i*3+2] = (Math.random() - .5) * 20;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    const pts = new THREE.Points(geo, new THREE.PointsMaterial({
      color: dark ? 0x7aa2ff : 0x3867f4, size: .07, transparent: true, opacity: dark ? .7 : .45 }));
    scene.add(pts);

    const rings = [];
    [[-12, 6, -6, 0x3867f4], [13, -5, -8, 0x12a594], [8, 8, -10, 0x3867f4]].forEach(([x, y, z, c]) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(2.2, .02, 8, 64),
        new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: dark ? .5 : .3 }));
      m.position.set(x, y, z);
      scene.add(m); rings.push(m);
    });

    const fit = () => {
      const w = P.innerWidth, h = P.innerHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    };
    P.addEventListener("resize", fit); fit();
    cleanups.push(() => P.removeEventListener("resize", fit));

    const mouse = { x: 0, y: 0 };
    const onMove = e => { mouse.x = e.clientX / P.innerWidth - .5; mouse.y = e.clientY / P.innerHeight - .5; };
    P.addEventListener("mousemove", onMove);
    cleanups.push(() => P.removeEventListener("mousemove", onMove));

    let stopped = false, raf;
    cleanups.push(() => { stopped = true; cancelAnimationFrame(raf); renderer.dispose(); });
    const clock = new THREE.Clock();
    const loop = () => {
      if (stopped) return;
      const t = clock.getElapsedTime();
      pts.rotation.y = t * .02;
      pts.rotation.x = Math.sin(t * .1) * .05;
      rings.forEach((r, i) => { r.rotation.x = t * (.15 + i * .05); r.rotation.y = t * (.2 - i * .04); });
      camera.position.x += (mouse.x * 2.2 - camera.position.x) * .03;
      camera.position.y += (-mouse.y * 1.4 - camera.position.y) * .03;
      camera.lookAt(0, 0, 0);
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    loop();
  } catch (e) { console.warn("Background scene disabled:", e); }

  /* ---------- 2. GSAP: reveal on scroll ---------- */
  const REVEAL = '[data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stExpander"], .block-container h3';
  const io = new P.IntersectionObserver(entries => {
    entries.forEach(en => {
      if (!en.isIntersecting) return;
      io.unobserve(en.target);
      gsap.to(en.target, { y: 0, opacity: 1, scale: 1, duration: .9, ease: "power3.out", clearProps: "transform,opacity" });
    });
  }, { threshold: 0.08 });
  cleanups.push(() => io.disconnect());

  /* ---------- 3. anime.js: count-up for metric values ---------- */
  function countUp(el) {
    const text = el.textContent.trim();
    if (el.__final === text || el.__running) return;
    el.__final = text;
    const m = text.match(/^([^0-9+\\-]*)([+\\-]?[0-9][0-9,]*\\.?[0-9]*)(.*)$/);
    if (!m || /^Store/i.test(text)) return;
    const raw = m[2].replace(/,/g, "");
    const target = parseFloat(raw);
    if (!isFinite(target)) return;
    const decimals = (raw.split(".")[1] || "").length;
    const state = { v: 0 };
    el.__running = true;
    anime({
      targets: state, v: target, duration: 1500, easing: "easeOutExpo",
      update: () => {
        const n = state.v.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
        el.textContent = m[1] + n + m[3];
      },
      complete: () => { el.textContent = text; el.__running = false; }
    });
  }

  /* ---------- 4. GSAP: 3D tilt on metric cards ---------- */
  let tilted = null;
  const onTilt = e => {
    const card = e.target.closest && e.target.closest('[data-testid="stMetric"]');
    if (tilted && tilted !== card) {
      gsap.to(tilted, { rotationX: 0, rotationY: 0, y: 0, scale: 1, duration: .6, ease: "elastic.out(1,.6)" });
      tilted = null;
    }
    if (!card) return;
    const r = card.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5;
    card.style.setProperty("--mx", (px + .5) * 100 + "%");
    card.style.setProperty("--my", (py + .5) * 100 + "%");
    gsap.to(card, { rotationY: px * 12, rotationX: -py * 12, y: -4, scale: 1.03, transformPerspective: 700, duration: .35, ease: "power2.out" });
    tilted = card;
  };
  D.addEventListener("mousemove", onTilt);
  cleanups.push(() => D.removeEventListener("mousemove", onTilt));

  /* ---------- 5. Back-to-top button (GSAP show/hide, anime.js scroll) ---------- */
  const css = D.createElement("style");
  css.id = "fx-btt-style";
  css.textContent = `
    #fx-btt { position: fixed; right: 28px; bottom: 28px; z-index: 999999; width: 52px; height: 52px;
      border: 0; border-radius: 50%; cursor: pointer; display: grid; place-items: center;
      background: linear-gradient(135deg, #3867f4, #12a594); color: #fff;
      box-shadow: 0 10px 28px rgba(56,103,244,.5); opacity: 0; pointer-events: none; }
    #fx-btt::after { content: ""; position: absolute; inset: -6px; border-radius: 50%;
      border: 2px solid rgba(56,103,244,.45); animation: fxPulse 2s ease-out infinite; }
    #fx-btt svg { width: 22px; height: 22px; }
    @keyframes fxPulse { from { transform: scale(.9); opacity: .9; } to { transform: scale(1.35); opacity: 0; } }
  `;
  D.head.appendChild(css);
  const btn = D.createElement("button");
  btn.id = "fx-btt";
  btn.type = "button";
  btn.setAttribute("aria-label", "Back to top");
  btn.title = "Back to top";
  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
  D.body.appendChild(btn);
  cleanups.push(() => { btn.remove(); css.remove(); });

  let scroller = null, shown = false;
  const onScroll = e => {
    const t = e.target;
    if (!t || t === D || !t.scrollTop && t.scrollTop !== 0) return;
    if (t.scrollHeight <= t.clientHeight + 50) return;
    scroller = t;
    const show = t.scrollTop > 400;
    if (show === shown) return;
    shown = show;
    gsap.to(btn, { opacity: show ? 1 : 0, y: show ? 0 : 20, scale: show ? 1 : .6,
                   duration: .4, ease: "back.out(1.7)", pointerEvents: show ? "auto" : "none" });
  };
  D.addEventListener("scroll", onScroll, true);
  cleanups.push(() => D.removeEventListener("scroll", onScroll, true));
  btn.addEventListener("click", () => {
    const el = scroller || D.querySelector('[data-testid="stMain"]') || D.querySelector("section.main");
    if (el) anime({ targets: el, scrollTop: 0, duration: 900, easing: "easeInOutCubic" });
  });
  btn.addEventListener("mouseenter", () => anime({ targets: btn, scale: 1.15, duration: 250, easing: "easeOutQuad" }));
  btn.addEventListener("mouseleave", () => anime({ targets: btn, scale: 1, duration: 400, easing: "easeOutElastic(1, .6)" }));

  /* ---------- Scan for new elements (runs on every Streamlit rerun) ---------- */
  function scan() {
    const fresh = [...D.querySelectorAll(REVEAL)].filter(el => !el.__fx);
    fresh.forEach(el => {
      el.__fx = true;
      gsap.set(el, { y: 34, opacity: 0, scale: .97 });
      io.observe(el);
    });
    D.querySelectorAll('[data-testid="stMetricValue"] div').forEach(countUp);

    const sb = D.querySelector('[data-testid="stSidebar"]');
    if (sb && !sb.__fx) {
      sb.__fx = true;
      gsap.from(sb.querySelectorAll('[data-testid="stSidebarUserContent"] > div > div, [data-testid="stSidebarContent"] [data-testid="stElementContainer"]'),
        { x: -26, opacity: 0, duration: .7, stagger: .07, ease: "power3.out", clearProps: "transform,opacity" });
    }
  }
  let timer;
  const mo = new P.MutationObserver(() => { clearTimeout(timer); timer = setTimeout(scan, 120); });
  mo.observe(D.body, { childList: true, subtree: true });
  cleanups.push(() => { mo.disconnect(); clearTimeout(timer); });
  scan();
})();
</script>
"""
components.html(
    FX_JS.replace("__DARK__", "true" if dark_mode else "false")
    .replace("0x3867f4", "0xff4b4b")
    .replace("0x12a594", "0x416aaf")
    .replace("#3867f4", COLORS["primary"])
    .replace("#12a594", COLORS["blue"])
    .replace("rgba(56,103,244", "rgba(255,75,75")
    .replace("rgba(18,165,148", "rgba(65,106,175")
    .replace("rgba(56, 103, 244", "rgba(255, 75, 75")
    .replace("rgba(18, 165, 148", "rgba(65, 106, 175"),
    height=0,
)