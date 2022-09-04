import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from pages.EDA import *
from pages.ML import *

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
    classical_AB()
    sequential_AB()
    ML_sequential()


def classical_AB():
    st.write("")
    st.header("Classical AB Testing")
    st.markdown("""
    In our classical A/B Testing we have performed the following steps
    """)
    st.markdown("""
    `*` Define the baseline conversion rate and minimum detectable effect (MDE).
    """)
    st.markdown("""
    `*` Calculate the sample size using statistical power and significance with the addition of the metrics in the above step.
    """)
    st.markdown("""
    `*` Drive traffic to your variations until you reach the target sample for each variation
    """)
    st.markdown("""
    `*` Finally, evaluate the results of your A/B test.
    """)
    st.markdown("""
    If the difference in performance between variations reached MDE or exceeded it,
     the hypothesis of your experiment is proven right, otherwise, it’s necessary to start the test from scratch.
    """)
    st.title("Limitations and Challenges of Classical A/B Testing")
    st.markdown("""
    `*` Can take lots of time and resources to collect the desired sample size.
    """)
    st.markdown("""
    `*` Can not handle multiple variable complex systems.
    """)


def sequential_AB():
    st.write("")
    st.header("Sequential AB Testing")
    st.markdown("""
    `*` Sequential A/B testing allows experimenters to analyze data
     while the test is running in order to determine if an early decision can be made.
    """)
    st.markdown("""
    `*` sequential sampling works in a very non-traditional way; instead of a fixed sample size, 
    you choose one item (or a few) at a time and then test your hypothesis. 
    """)
    st.markdown("""
    `*` We will use the Sequential probability ratio testing (SPRT) algorithm,
     which is based on the likelihood ratio statistic, for our dataset.
    """)
    st.markdown("""
    General steps of conditional SPRT
    """)
    st.markdown("""
    `1. ` Calculate critical upper and lower decision boundaries
    """)
    st.markdown("""
    `2. ` Perform cumulative sum of the observation
    """)
    st.markdown("""
    `3. ` Calculate test statistics(likelihood ratio) for each of the observations
    """)
    st.markdown("""
    `4. ` Calculate upper and lower limits for the exposed group
    """)

def ML_sequential():
    st.write("")
    st.header("Significance Testing Using Machine Learning")
    st.markdown("""
    It’s not just a matter of calculating the difference between the exposed and controlled variances to determine which one has been chosen when using machine learning for A/B testing;
     instead, it’s about figuring out which parameter 
    (variable) in the data has the highest significance value for predicting the outcome.
    """)
    st.markdown("""
    In order to conduct this experiment, we assembled 3machine learning models
     of our choosing and attempted to determine each model’s accuracy score and correlation matrix.
    """)
    st.title("Five-fold cross-validation")
    st.markdown("""
    `*` The ﬁve-fold cross-validation (CV) is a process when all data is randomly split into k folds, in our casek=5, and then the model is 
    trained on the k−1 folds, while one fold is left to test a model.
    """)
    st.markdown("""
    `*` This procedure is repeated k times. However,in this work, all data ﬁrst is split into 
    training and testingdatasets, and a training dataset is used for cross-validation.
    """)

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