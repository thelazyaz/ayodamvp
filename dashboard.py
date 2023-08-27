import streamlit as st
import pickle
import pandas as pd

st.image('Text Logo.png')
st.markdown("Reducing your waste down to an iota")

model = pickle.load(open('OrdersForecaster.pkl', "rb"))

purchases = pd.read_excel('Data Model - Pizza Sales.xlsx')

st.markdown("# Your Data")
st.markdown("## Number of Orders in 2015")
purchasesperday = purchases.order_date.value_counts()[purchases.order_date.unique()]
st.line_chart(purchasesperday)