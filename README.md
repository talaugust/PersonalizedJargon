# README

This is the repository for the paper: **Personalized Jargon Identification for Enhanced Interdisciplinary Communication**.

_Abstract_: Scientific jargon can impede researchers when they read materials from other domains. Current methods of jargon identification mainly use corpus-level familiarity indicators (e.g., Simple Wikipedia represents plain language). However, researchers' familiarity of a term can vary greatly based on their own background. We collect a dataset of over 10K term familiarity annotations from 11 computer science researchers for terms drawn from 100 paper abstracts. Analysis of this data reveals that jargon familiarity and information needs vary widely across annotators, even within the same sub-domain (e.g., NLP). We investigate features representing individual, sub-domain, and domain knowledge to predict individual jargon familiarity. We compare supervised and prompt-based approaches, finding that prompt-based methods including personal publications yields the highest accuracy, though zero-shot prompting provides a strong baseline. This research offers insight into features and methods to integrate personal data into scientific jargon identification.


Eventually this repository will contain the data, models, and analyses from the paper. For now, we have structured the README to explain this information.

## Data

The `data/` directory currently contains the out-of-domain abstracts used in our annotation task. The data was collected in the following way: 
> To ensure that the out-of-domain abstracts could realistically be read by our annotators, we compile a corpus of non-CS papers often viewed by CS researchers, published after 2010, using the [Semantic Scholar API](https://www.semanticscholar.org/product/api). We define CS researchers as anyone who has co-authored a paper categorized as `Computer Science' using the API. We take the top 500 viewed papers not categorized as CS.   

## Models 

## Analyses 
