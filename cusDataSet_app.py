import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly

from db_connect import DatabaseConnectionError, fetch_walmart_sales


st.set_page_config(page_title="Walmart Sales Forecast", layout="wide")
st.title("Walmart Sales Forecast")


@st.cache_data(show_spinner="Loading sales data from MySQL...")
def load_sales_data() -> pd.DataFrame:
    sales = fetch_walmart_sales()
    sales["Date"] = pd.to_datetime(
        sales["Date"], dayfirst=True, errors="raise"
    ).dt.normalize()
    return sales.sort_values(["Date", "Store"]).reset_index(drop=True)


@st.cache_data
def prepare_forecast_data(sales: pd.DataFrame) -> pd.DataFrame:
    return (
        sales.groupby("Date", as_index=False)[["Weekly_Sales"]]
        .sum()
        .rename(columns={"Date": "ds", "Weekly_Sales": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner="Training Prophet on weekly sales...")
def train_forecast_model(training_data: pd.DataFrame) -> Prophet:
    model = Prophet(
        interval_width=0.8,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    model.fit(training_data)
    return model


try:
    sales_data = load_sales_data()
except DatabaseConnectionError as error:
    st.error(str(error))
    st.stop()

if sales_data.empty:
    st.error("The MySQL `walmart_sales` table contains no records.")
    st.stop()

forecast_data = prepare_forecast_data(sales_data)
if len(forecast_data) < 2:
    st.error("At least two weekly sales records are needed to train a forecast.")
    st.stop()

with st.expander("Source data and summary", expanded=True):
    st.subheader("Database records")
    st.caption(f"{len(sales_data):,} rows loaded from `cusNew.walmart_sales`.")
    sales_display = sales_data.copy()
    sales_display["Date"] = sales_display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(sales_display, width="stretch")

    summary_col, missing_col = st.columns(2)
    with summary_col:
        st.subheader("Descriptive statistics")
        st.dataframe(sales_data.describe(), width="stretch")
    with missing_col:
        st.subheader("Missing values by column")
        missing_values = sales_data.isnull().sum().rename("Missing values")
        st.dataframe(missing_values, width="stretch")

st.subheader("Weekly total sales")
st.caption("Weekly sales are summed across all stores, matching the notebook training workflow.")
st.dataframe(forecast_data, width="stretch")

periods = st.sidebar.slider("Forecast horizon (weeks)", 4, 52, 12, step=4)
model = train_forecast_model(forecast_data)
future_dates = model.make_future_dataframe(periods=periods, freq="W-FRI")
forecast = model.predict(future_dates)
future_forecast = forecast.loc[forecast["ds"] > forecast_data["ds"].max()]

st.subheader(f"{periods}-week sales forecast")
forecast_figure = go.Figure()
forecast_figure.add_trace(
    go.Scatter(
        x=forecast_data["ds"],
        y=forecast_data["y"],
        mode="lines",
        name="Actual weekly sales",
        line={"color": "#17233c", "width": 2},
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat_upper"],
        mode="lines",
        line={"width": 0},
        showlegend=False,
        hoverinfo="skip",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(56, 103, 244, 0.18)",
        line={"width": 0},
        name="80% prediction interval",
        hoverinfo="skip",
    )
)
forecast_figure.add_trace(
    go.Scatter(
        x=future_forecast["ds"],
        y=future_forecast["yhat"],
        mode="lines",
        name="Forecast",
        line={"color": "#3867f4", "width": 3, "dash": "dash"},
    )
)
forecast_figure.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="Weekly sales",
    hovermode="x unified",
)
st.plotly_chart(forecast_figure, width="stretch")

st.subheader("Forecast values")
forecast_display = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
    columns={
        "ds": "Date",
        "yhat": "Predicted sales",
        "yhat_lower": "Lower bound",
        "yhat_upper": "Upper bound",
    }
)
st.dataframe(forecast_display, width="stretch")
st.download_button(
    "Download forecast as CSV",
    forecast_display.to_csv(index=False).encode("utf-8"),
    file_name="walmart_sales_forecast.csv",
    mime="text/csv",
)

with st.expander("Trend and seasonality components"):
    st.plotly_chart(plot_components_plotly(model, forecast), width="stretch")
