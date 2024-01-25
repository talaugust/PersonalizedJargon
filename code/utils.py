import pandas as pd
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

from collections import Counter

import os
import json
import numpy as np
stop_words = set(stopwords.words('english'))

import openai
from tenacity import retry, wait_chain, wait_fixed
@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
            [wait_fixed(3) for i in range(2)] +
            [wait_fixed(10)]))
def chat_complition_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def is_alphabet_or_hyphen(word):
    return bool(re.match(r'^[a-zA-Z-]+$', word))

def compute_word_frequency(full_text_list):
    """
    Compute word frequency from a list of text segments.
    
    Parameters:
    - full_text_list (list of str): List of text segments.
    - output_name (str): Name of the output CSV file.
    
    Returns:
    - None. Saves the word frequency to a CSV file.
    """
    # Filter out None values or empty strings
    full_text_list = [text for text in full_text_list if text]
    stop_words = set(stopwords.words('english'))
    
    # Tokenize words and remove stopwords
    words = []
    for text in full_text_list:
        try:
            tokenized = word_tokenize(text)
            words.extend([word.lower() for word in tokenized if is_alphabet_or_hyphen(word) and word.lower() not in stop_words])  # Consider alphanumeric words only and exclude stopwords
        except:
            continue
        # tokenized = word_tokenize(text.lower())
            
    # Compute word frequency
    word_freq = Counter(words)
    
    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(word_freq.items(), columns=['word', 'frequency'])
    df = df.sort_values(by='frequency', ascending=False)
    return df

def compute_word_frequency_for_entity_in_target(entity, paper_sha, target_paper_path):
    tmp = word_tokenize(entity)
    entity_list = []
    entity_list.extend([word.lower() for word in tmp if is_alphabet_or_hyphen(word) and word.lower() not in stop_words])
    paper_path = os.path.join(target_paper_path, paper_sha + '.json')
    with open(paper_path) as f:
        paper_info = json.load(f)
    abstract = paper_info['abstract']
    title = paper_info['title']
    if abstract is None:
        abstract = ''
    if title is None:
        title = ''
    title_abstract = title + ' ' + abstract

    frequency_abstract_df = compute_word_frequency([title_abstract])
    return frequency_abstract_df, entity_list

def compute_tf_idf(full_text_list, output_name):
    # Filter out None values or empty strings
    full_text_list = [text for text in full_text_list if text]

    # Check if the list is empty
    if not full_text_list:
        return

    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform(full_text_list)
        feature_names = vectorizer.get_feature_names_out()
        dense = X.todense()
        denselist = dense.tolist()

        df_tfidf = pd.DataFrame(denselist, columns=feature_names)
        df_tfidf = df_tfidf.loc[:, ~df_tfidf.columns.str.isnumeric()]

        # Remove words starting with numbers and create a mapping from original to cleaned words
        cleaned_word_map = {word: re.sub(r'^\d+', '', word) for word in df_tfidf.columns}
        
        # Group columns by the cleaned word and sum
        df_tfidf = df_tfidf.groupby(cleaned_word_map, axis=1).sum()

        # Save the words and their TF-IDF values to a CSV file
        df_tfidf_avg = df_tfidf.mean(axis=0).reset_index()
        df_tfidf_avg.columns = ['word', 'tfidf']
    except ValueError as e:
        print(f"Error with {output_name}: {e}")
    return df_tfidf_avg


def cal_specificity(frequency_1, frequency_2):
    if frequency_2 == 0 or frequency_1 == 0:
        return -float('inf')
    return math.log2(frequency_1 / frequency_2)

def cal_zscore(df):
    annotator_list = df['annotator'].unique().tolist()
    for annotator in annotator_list:
        # calculate z-score of familiarity among the annotator
        df.loc[df['annotator'] == annotator, 'familiarity_zscore_annotator'] = (df.loc[df['annotator'] == annotator, 'familiarity'] - df.loc[df['annotator'] == annotator, 'familiarity'].mean()) / df.loc[df['annotator'] == annotator, 'familiarity'].std(ddof=0)
    return df

def fit_linear_regression(x, y):
    """
    Fit a linear regression model. Return the slope of the model (i.e., the coefficient of x). Also return the p-value of the slope.
    
    Parameters:
    - x (list of float): List of x values.
    - y (list of float): List of y values.
    
    Returns:
    - (float): Slope of the linear regression model.
    - (float): P-value of the slope.
    """
    import statsmodels.api as sm
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results.params[1], results.pvalues[1].round(3)

def get_specter2_embeddings(text_batch, tokenizer, model, device):
    # Ensure the model is on the correct deviceßß
    model = model.to(device)
    
    inputs = tokenizer(text_batch, padding=True, truncation=True,
                                    return_tensors="pt", return_token_type_ids=False, max_length=512)
    
    # Move the inputs to the correct device
    for key, tensor in inputs.items():
        inputs[key] = tensor.to(device)
    
    output = model(**inputs)
    embeddings = output.last_hidden_state[:, 0, :]
    # detach from the graph and move to cpu
    embeddings = embeddings.detach().cpu()
    return embeddings

def get_response_message(response):
    return response['choices'][0]['message']['content']

def chat_additional_info(instructions, model_name='gpt-4', category='additional_definition'):

    if category == 'additional_definition':
        key = 'additional Definition/Explanation'
        definition = 'Definition of definition/explanation: provides key information on the term independent of any context (e.g., a specific scientific abstract). A definition answers the question, \"What is/are [term]?\"'
    elif category == 'additional_background':
        key = 'additional Background/Motivation'
        definition = 'Definition of background/motivation: introduces information that is important for understanding the term in the context of the abstract. Background can provide information about how the term relates to overall problem, significance, and motivation of the abstract.'
    elif category == 'additional_example':
        key = 'additional Example'
        definition = 'Definition of example: offers specific instances that help illustrate the practical application or usage of the term within the abstract.'
    prefix_text = ''

    messages = [
    {"role": "system", "content": f"You are tasked with predicting whether the reader might need {key} to fully grasp the entities mentioned in a given abstract. You will be provided with the entity list, the abstract where the entities come from, and related data pertinent to the reader or the abstract. {definition}"},
    {"role": "user", "content": instructions},
    ]
    response = chat_complition_with_backoff(
        # model="gpt-3.5-turbo-16k-0613",
        model=model_name,
        messages=messages,
        max_tokens=100,
    )
    response_content = get_response_message(response)
    return response_content

def generate_prompt_additional_info(entity_list, abstract, related_data):
    prompt = """
            Entities: 
            {}
            Abstract:
            {}
            Related data:
            {}
            Provide the prediction whether additional information is needed in a list in the order of the entity. The prediction should be either 0(no) or 1(yes). No need to mention the entity:
            """.format(entity_list, abstract, related_data)
    
    return prompt

def extract_related_data(df_domain, example_num):
    # drop rows with empty abstract
    df_domain = df_domain.dropna(subset=['abstract', 'paper_name'])
    df_domain = df_domain.reset_index(drop=True)
    df_domain_se = df_domain.sample(example_num, random_state=1)
    abstracts = df_domain_se['abstract'].tolist()
    titles = df_domain_se['paper_name'].tolist()
    for i in range(len(abstracts)):
        if type(abstracts[i]) != str:
            abstracts[i] = ''
        if type(titles[i]) != str:
            titles[i] = ''
    abstracts = [title + ' ' + abstract for title, abstract in zip(titles, abstracts)]
    return abstracts

def get_top_k_indices(scores, k):
    sorted_indices = list(np.argsort(scores)[::-1])
    top_k_indices = sorted_indices[:k]
    
    while len(top_k_indices) < len(scores) and scores[top_k_indices[-1]] == scores[sorted_indices[len(top_k_indices)]]:
        top_k_indices.append(sorted_indices[len(top_k_indices)])
        
    return set(top_k_indices)

def top_k_recall_precision(true_scores, predicted_scores, k):
    true_top_k = get_top_k_indices(true_scores, k)
    predicted_top_k = get_top_k_indices(predicted_scores, k)
    
    intersect_count = len(true_top_k.intersection(predicted_top_k))
    
    recall = intersect_count / len(true_top_k)
    precision = intersect_count / len(predicted_top_k)
    
    return recall, precision

def chat_binary(instructions, model_name='gpt-4'):
    messages = [
    {"role": "system", "content": "Your job is to estimate how much the reader knows about an entity. You will be provided with the entity, the abstract where the entity come from, and additional information about either the reader or the abstract. "},
    {"role": "user", "content": instructions},
    ]
    # response = openai.ChatCompletion.create(
    response = chat_complition_with_backoff(
        model=model_name,
        messages=messages,
        max_tokens=100,
    )
    response_content = get_response_message(response)
    return response_content
    

def generate_prompt_binary(entity_list, abstract, related_data):
    prompt = """
            Entity: {}
            Abstract:{}
            Additional information:{}
            Here's how to gauge the reader's familiarity: - 0: The reader knows this subject well and can describe it to others. - 1: The reader has either encountered this subject before but knows little about it, or has never come across it at all.
            Based on the information provied, determine the familiarity score, either 0 or 1:
            """.format(entity_list, abstract, related_data)
    
    return prompt

def chat_additional_info_binary(instructions, model_name='gpt-4', category='additional_definition'):

    if category == 'additional_definition':
        key = 'additional Definition/Explanation'
        definition = 'Definition of definition/explanation: provides key information on the term independent of any context (e.g., a specific scientific abstract). A definition answers the question, \"What is/are [term]?\"'
    elif category == 'additional_background':
        key = 'additional Background/Motivation'
        definition = 'Definition of background/motivation: introduces information that is important for understanding the term in the context of the abstract. Background can provide information about how the term relates to overall problem, significance, and motivation of the abstract.'
    elif category == 'additional_example':
        key = 'additional Example'
        definition = 'Definition of example: offers specific instances that help illustrate the practical application or usage of the term within the abstract.'
    prefix_text = ''

    messages = [
    {"role": "system", "content": f"Your job is to estimate whether the reader might need {key} to fully grasp the entities mentioned in a given abstract. You will be provided with the entity, the abstract where the entity come from, and additional information about either the reader or the abstract. {definition}"},
    {"role": "user", "content": instructions},
    ]
    response = chat_complition_with_backoff(
        # model="gpt-3.5-turbo-16k-0613",
        model=model_name,
        messages=messages,
        max_tokens=100,
    )
    response_content = get_response_message(response)
    return response_content

def generate_prompt_additional_info_binary(entity_list, abstract, related_data):
    prompt = """
            Entities: {}
            Abstract: {}
            Related data: {}
            Provide the estimation whether additional information is needed in a list in the order of the entity. The estimation should be either 0(no) or 1(yes). No need to mention the entity:
            """.format(entity_list, abstract, related_data)
    
    return prompt