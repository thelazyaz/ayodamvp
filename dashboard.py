import streamlit as st
import pickle
import pandas as pd

st.image('Text Logo.png')
st.markdown("Reducing your waste down to an iota")

model = pickle.load(open('OrdersForecaster.pkl', "rb"))

uploaded_file = st.file_uploader("Upload Your Custom Resturant Data")
if st.button("Submit", type="primary"):

    purchases = pd.read_csv(uploaded_file) #pd.read_csv('Data Model - Pizza Sales.csv')

    st.markdown("# Your Data")

    st.markdown("## Overview")
    data=[
                purchases["pizza_name"].value_counts().index.to_list()[:5],
                purchases["pizza_name"].value_counts().to_list()[:5],
                purchases["pizza_ingredients"].value_counts().index.to_list()[:5],
                purchases["pizza_category"].value_counts().index.to_list()[:5],
                purchases["pizza_size"].value_counts().index.to_list()[:5],
        ]
    st.table(pd.DataFrame(data, columns=list(range(1,6)),
                index=["Top 5 Pizzas",
                        "Orders",
                        "Ingredients",
                        "Pizza Category",
                        "Pizza Size"
                        ]))

    st.markdown("## Number of Orders in 2015")
    purchasesperday = purchases.order_date.value_counts()[purchases.order_date.unique()].astype(int)
    print(purchasesperday)
    st.line_chart(purchasesperday)

    st.markdown("## Percent Quantity Pizzas Bought out of total")
    st.table(pd.DataFrame([round(p/sum(purchases["quantity"].value_counts().to_list())*100, 2) for p in purchases["quantity"].value_counts().to_list()],
                columns =[""], index=list(range(1,5))))

    st.markdown("## Top 5 most popular times to buy pizza")
    ordertimes=purchases["order_time"].value_counts().to_list()[:5]
    st.table(pd.DataFrame([purchases["order_time"].value_counts().index.to_list()[:5], ordertimes], columns=list(range(1,6)), index=["", "Orders bought"]))

    X=purchasesperday.tail(-1).to_numpy()
    y=purchasesperday.shift(1).fillna(0).tail(-1).to_numpy()
    #for now just total orders, we can do orders of each pizza later

    X_train = X.reshape(-1,1)[:300]
    y_train = y[:300]
    X_test = X.reshape(-1,1)[300:]
    y_test = y[300:]

    st.markdown("## Model Predictions")
    st.markdown("### Actual vs Predicted Orders for last 2 months of the year")
    predictions = model.predict(X_test)
    st.line_chart(y_test)
    st.line_chart(predictions)
