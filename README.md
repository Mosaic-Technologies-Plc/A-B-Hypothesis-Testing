# A/B Hypothesis Testing - 10 Academy Trianing Batch 6 Week 2 Challenge

***

**Table of Contents**

- [A/B Hypothesis Testing](#A-B-Hypothesis-Testing)
  - [Overview](#overview)
  - [About](#about)
  - [About The Data](#about-data)
  - [Project Structure](#project-structure)
    - [.github](#.github)
    - [data](#data)
    - [notebooks](#notebooks)
    - [scripts](#scripts)
    - [root folder](#root-folder)

***

## Overview
An advertising company is running an online ad for a client with the intention of increasing brand awareness. The advertiser company earns money by charging the client based on user engagements with the ad it designed and serves via different platforms. To increase its market competitiveness, the advertising company provides a further service that quantifies the increase in brand awareness as a result of the ads it shows to online users. The main objective of this project is to test if the ads that the advertising company runs resulted in a significant lift in brand awareness. 

## About
SmartAd is a mobile first advertiser agency. It designs intuitive touch-enabled advertising. It provides brands with an automated advertising experience via machine learning and creative excellence. Their company is based on the principle of voluntary participation which is proven to increase brand engagement and memorability 10 x more than static alternatives. SmartAd provides an additional service called Brand Impact Optimiser (BIO), a lightweight questionnaire, served with every campaign to determine the impact of the creative, the ad they design, on various upper funnel metrics, including memorability and brand sentiment.

As a Machine learning engineer in SmartAd, the task was to design a reliable hypothesis testing algorithm for the BIO service and to determine whether a recent advertising campaign resulted in a significant lift in brand awareness.

![Alt text](MLFlow.png?raw=true "ML Flow")

## About The Data
The BIO(Brand Impact Optimizer) data for this project is a "Yes" or "No" response from online users to the below question
- Q: Do you know the brand Lux?
    - [ ] Yes
    - [ ] No


## Project Structure
The repository has a number of files including python scripts, jupyter notebooks, raw and cleaned data, and text files. Here is their structure with a brief explanation.

### .github
- a configuration file for github actions and workflow
- `workflows/CI.yml` continous integration configuration

### data
- the folder where the raw, and cleaned datasets' csv files are stored

### notebooks
- `task1_2_EDA.ipynb`: a jupyter notebook that contains EDA for our dataset
- `classical AB testing.ipynb`: a jupyter notebook that performs classical AB testing on our dataset
- `data_cleaning.ipynb`: a jupyter notebook that handles cleaning before EDA can be performed 
- `platform and browser splitter.ipynb`: a jupyter notebook that handle data versioning for each browser and platform OS
- `sequential AB testing.ipynb`: a jupyter notebook that performs sequential AB testing on our dataset

### scripts

### root folder
- `requirements.txt`: a text file lsiting the projet's dependancies
- `.gitignore`: a text file listing files and folders to be ignored
- `.dvcignore`: .ignore file for `dvc`
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.
