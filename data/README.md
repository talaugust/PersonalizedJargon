# Dataset Folder

The dataset folder contains two files:

1. `annotation_results_meta_data.csv`: This file contains all the annotation results for the entities extracted from the abstracts that need to be personalized.

## Abstract Details

- `entity`: The entity extracted from the abstract to be personalized.
- `paper_sha`: The paper ID in the Semantic Scholar (S2) API.
- `abstract`: The full abstract text.
- `s2fieldofstudy`: The field of study of the abstract.

## Annotation Results

1. **Familiarity**: Annotators were asked to rate their familiarity with the term on a scale of 1 to 5, where 1 indicates "Not at all Familiar" and 5 indicates "Extremely Familiar."
   - `familiarity`: The familiarity rating after reading the abstract. This is the familiarity used in the paper results.
   - `familiarity_before`: The familiarity rating before reading the abstract.

2. **Additional Information Needs**: Detailed questions can be found in `./term-familiarity-survey-generator`.
   - `additional_definition`: Indicates whether the annotator needs an additional definition for the entity.
   - `additional_background`: Indicates whether the annotator needs additional background information for the entity.
   - `additional_example`: Indicates whether the annotator needs an additional example for the entity.

## Annotator Metadata

- `annotator`: Anonymized annotator ID from 1-11. This aligns with the annotator ID in the paper.
- `self-reported domain`: The annotator's self-reported subdomain, such as networking, NLP, CV, etc.
- `number_of_papers`: The number of papers published by the annotator.
- `reference_count`: The average number of references in the papers published by the annotator.
- `first_paper_year`: The year of the first published paper by the annotator.
- `first_cs_paper_year`: The year of the first computer science paper published by the annotator. The domain of the paper is determined by the paper's `s2fieldofstudy`.

2. `top_non_cs_papers_ids.csv`: To ensure that the out-of-domain abstracts could realistically be read by our annotators, we compile a corpus of non-CS papers often viewed by CS researchers, published after 2010, using the [Semantic Scholar API](https://www.semanticscholar.org/product/api). We define CS researchers as anyone who has co-authored a paper categorized as 'Computer Science' using the API. This file contains the top 500 viewed papers not categorized as CS.