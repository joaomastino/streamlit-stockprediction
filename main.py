import streamlit as st
from datetime import date

import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("MT.MI", "LVMH.MI", "KER.PA", "PST.MI", "9638.HK", "TIT.MI", "FM.MI", "ISP.MI", "MONC.MI", "TGYM.MI", "CPR.MI", "ENV.MI", "BZU.MI")
select_stocks = st.selectbox("Seleziona il dataset per la predizione", stocks)

n_years = st.slider("Anni di predizione:", 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Sto caricando i dati...")
data = load_data(select_stocks)
data_load_state.text("Carico i dati... fatto!")

st.subheader('I dati in tabella')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Data'], y=data['Apertura'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Data'], y=data['Chiusura'], name='stock_close'))
    fig.layout.update(title_text="I dati nel tempo", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Dati predittivi')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)