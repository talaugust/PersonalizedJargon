import sys
sys.path.append("/alternative_abstract/")
from util.utils import *

import pandas as pd
import os
import openai
from tqdm import tqdm
import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='/results/binary_familiarity/gpt/')
    parser.add_argument('--data_path', type=str, default='/data/annotation_data/binary_classification/')
    parser.add_argument('--prefix', type=str, default='train_test_balance.csv')
    parser.add_argument('--api_key', type=str, default='')

    args = parser.parse_args()

    openai.api_key = args.api_key
    model_name = 'gpt-4'

    descrip_dict = {
        'self_defined_subfield': 'Self-defined subfield of the reader is:',
        'number_of_papers': 'Number of papers published by the reader is:',
        'reference_count': 'Number of papers referenced by the reader is:',
        'first_paper_year': 'Year of the first paper published by the reader is:',
        's2fieldsofstudy': 'Domain of study of the paper is:'
    }

    train_path = args.data_path + 'train/'
    test_path = args.data_path + 'test/'
    test_type = args.prefix

    annotator_list = [i.split('_')[0] for i in os.listdir(train_path)]
    unique_annotators = list(set(annotator_list))
    unique_annotators = unique_annotators

    output_df = pd.DataFrame()
    for annotator in tqdm(unique_annotators):
        df_test = pd.read_csv(test_path + annotator + '_' + test_type)
        empty_lists_dict = {key: [] for key in descrip_dict.keys()}
        empty_lists_dict['fam_meta_all'] = []
        for i in range(len(df_test)):
            abstract = df_test.iloc[i]['abstract']
            entity = df_test.iloc[i]['entity']
            for key in descrip_dict.keys():
                related_data = descrip_dict[key] + str(df_test.iloc[i][key])
                full_prompt = generate_prompt_binary(entity, abstract, related_data)
                response = chat_binary(full_prompt, model_name=model_name)
                empty_lists_dict[key].append(response)

            related_data = ''
            for key in descrip_dict.keys():
                related_data += descrip_dict[key] + str(df_test.iloc[i][key]) + '\n'
            full_prompt = generate_prompt_binary(entity, abstract, related_data)
            response = chat_binary(full_prompt, model_name=model_name)
            empty_lists_dict['fam_meta_all'].append(response)

        for key in empty_lists_dict.keys():
            new_name = 'fam_meta_' + key
            df_test[new_name] = empty_lists_dict[key]
        output_df = output_df.append(df_test)
    output_df.to_csv(args.output_path + 'gpt_baseline_meta.csv', index=False)


if __name__ == '__main__':
    main()
