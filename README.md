# Personalized Jargon Identification

This repository contains data and models related to the paper: [Personalized Jargon Identification for Enhanced Interdisciplinary Communication](https://arxiv.org/abs/2311.09481).

**Authors**: Yue Guo, Joseph Chee Chang, Maria Antoniak, Erin Bransom, Trevor Cohen, Lucy Lu Wang, and Tal August

To encourage researchers who are interested in working on personalization tasks, we release the dataset, questionnaires we used to collect the dataset, and model details in this repository.

## Data

To protect the annotators' privacy, we currently only release the annotation data and annotators' metadata. Details can be found in `./data`. If you're interested in other features used in the paper, please contact yguo50@uw.edu.

## Term Familiarity Survey Generator

To facilitate researchers who are interested in applying these settings to researchers in other domains, we make public the questionnaires used in the current study. Details can be found in `./term-familiarity-survey-generator`.

## Models

In the study, we applied the following models to predict the familiarity and additional information needs for the terms in the abstract:

- Majority baseline
- Lasso regression
- Nearest-neighbor classifier
- GPT

Prompts used in GPT can be found in `./code`.

