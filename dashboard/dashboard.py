import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from  pages.EDA import *
from  pages.ML import *

st.set_page_config(page_title="Dashboard", layout="wide")

 

def mainPage():
    st.title("Overview of AB Testing")
    st.write("")
    st.header("What is AB Testing")
    st.markdown("""
    `*`An A/B test is an experiment where you test two variants, A and B, 
    against each other to evaluate which one performs better in a randomized experiment.
    """)
    st.markdown("""
    `*`The two variants to be tested can be classified as; control group, 
    those who are shown the current state of a product or service and treatment group, those who are shown the testing product or service.
    """)
    st.markdown("""
    `*` The conversion rates for each group is then monitored to determine which one is better. 
    Randomness in A/B testing is key to isolate the impact of the change made and to reduce the potential impact of confounding variables.
    """)


    st.write("")
    st.header("Classical AB Testing")


def EDA():
    st.sidebar.markdown("# EDA ❄️")
    st.title("EDA")
    setEDATitle()

def ML():
    st.sidebar.markdown("# Machine Learning ❄️")
    st.title("Machine Learning")
    setMLTitle() 



page_names_to_funcs = {
    "Overview": mainPage,
    "EDA": EDA,
    "Machine Learning": ML,
} 

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()