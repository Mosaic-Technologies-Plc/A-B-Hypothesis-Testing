import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px


@st.cache
def loadData():
    df = pd.read_csv('data/AdSmartClean_data.csv')
    return df


def controlExposedPie(df):
    st.title("Control and Exposed Groups")
    
    df_count = df['experiment'].value_counts()
    fig = px.pie(df_count.head(15), values='experiment', names='experiment', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    colB1, colB2 = st.columns([2.5, 1])

    with colB1:
        st.plotly_chart(fig)
    with colB2:
        st.write(df_count)


def device_makes(df):
    st.title("Top 15 Device Makes")

    df_count = pd.DataFrame({'count': df.groupby(['device_make'])['auction_id'].count()}).reset_index()
    # df_count['Generic Smartphone']['count'] = 1
    df_count = df_count.sort_values("count", ascending=False)
    fig = px.pie(df_count[1:15], values='count', names='device_make', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # colB1, colB2 = st.columns([2.5, 1])

    # with colB1:
    st.plotly_chart(fig)
    # with colB2:
    st.write(df_count)

def browsers(df):
    st.title("Top 5 Browsers")

    df_count = pd.DataFrame({'count': df.groupby(['browser'])['auction_id'].count()}).reset_index()
    # df_count['Generic Smartphone']['count'] = 1
    df_count = df_count.sort_values("count", ascending=False)
    fig = px.pie(df_count.head(5), values='count', names='browser', width=500, height=350)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # colB1, colB2 = st.columns([2.5, 1])

    # with colB1:
    st.plotly_chart(fig)
    # with colB2:
    st.write(df_count)

def setEDATitle():
    st.markdown("`*` EDA for the initial dataset")
    df = loadData()
    controlExposedPie(df)
    device_makes(df)
    browsers(df)
    with st.expander("EDA for the new dataset"):
        controlExposedPieNewDS()

def controlExposedPieNewDS():
    st.markdown("`*` EDA for the new dataset")