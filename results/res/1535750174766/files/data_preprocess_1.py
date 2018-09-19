#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-13 下午8:37
# @Author  : Futai

import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
import os



def data_cleanse(file_path):
    '''
    :param table_file_path:
    :return: pandas.DataFrame
    '''

    # convert columns ROC10_days, ROC20_days, ROC50_days, ROC150_days, MACD_12_26, PO52H, PO52L to percent sign
    df = pd.read_csv(file_path, index_col=0)
    df.columns = map(str.lower, df.columns)

    df = df.drop(['trix_15_days'], 1)
    df = df.rename(index=str, columns={'std_10': 'std10_days', 'std_21': 'std21_days', 'macd_12_26': 'macd_12_26_days'})
    percent_list = ['roc10_days', 'roc20_days', 'roc50_days', 'roc150_days', 'macd_12_26_days', 'po52h', 'po52l']

    for indicator in percent_list:
        df[indicator] = df[indicator].apply(lambda x: x * 100)

    df = df.round(2)
    df['ticker'] = df['ticker'].str.lower()
    df['company_name'] = df['company_name'].str.lower()

    df['text'] = df['text'].apply(text_cleanse)
    df['text'] = df['text'].str.lower()

    df_train, df_validate, df_test = train_validate_test_split(df)
    return df, df_train, df_validate, df_test

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


def text_cleanse(text):
    text = text.replace('%', ' %')
    text = text.replace('. ', ' . ')
    text = text.replace(', ', ' , ')

    return text

def create_table_summary(file_path, table_file, summary_file):
    '''

    :param file_path: the original datasets file path
    :return: txt1: txt file contains only table information; txt2: txt file contains only summary information
    '''
    df = pd.read_csv(file_path)
    df_table = df[['ticker', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume', 'sma_10_days', 'sma_20_days',
                   'sma_50_days', 'sma_150_days', 'ema_12_days', 'ema_26_days', 'roc10_days', 'roc20_days', 'roc50_days',
                   'roc150_days', 'atr_14_days', 'std10_days', 'std21_days', 'rsi_14_days', 'macd_12_26_days', 'po52h',
                   'po52l']]
    df_summary = df['text']

    table_dic = df_table.to_dict(orient='records')


    # generate table text file
    with open('original_data/' + table_file, 'w') as f:
        for i in range(len(table_dic)):
            for key, value in table_dic[i].items():
                if isinstance(value, float):
                    if value < 0:
                        value = abs(value)
                        f.write('{0}_1:- {0}_2:{1} '.format(key, value))
                    else:
                        f.write('{0}:{1} '.format(key, value))
                elif key == 'company_name':
                    content = value.split()
                    for i in range(len(content)):
                        f.write('{0}_{1}:{2} '.format(key, i+1, content[i]))
                else:
                    f.write('{0}:{1} '.format(key, value))
            f.write('\n')

    with open('original_data/' + summary_file, 'w') as f:
        for i in range(len(df_summary)):
            f.write('{0}'.format(df_summary[i]))
            f.write('\n')

def create_vocab(file_path, max_words_length=None):
    df = pd.read_csv(file_path, index_col=0)
    contents = []
    features = []
    for i, row in df.iterrows():
        feature_row = []
        content_row = []
        for key, value in row.iteritems():
            key = key.lower()
            value = str(value)
            if key == 'date':
                content_row.append(value)
                feature_row.append(key)
            elif key == 'company_name':
                value_list = value.split()
                for j in value_list:
                    content_row.append(j)
                    feature_row.append(key)
            elif key == 'text':
                value_list = value.split()
                content_row += value_list
            else:
                try:
                    if '-' in value:
                        value = value.replace('-', '- ')
                        content_row_1 = value.split()[0]
                        content_row_2 = value.split()[1]
                        content_row.append(content_row_1)
                        content_row.append(content_row_2)
                        feature_row.append(key)
                        feature_row.append(key)
                    else:
                        content_row.append(value)
                        feature_row.append(key)
                except:
                    content_row.append(value)
                    feature_row.append(key)

        contents.append(content_row)
        features.append(feature_row)

    flat_contents = []
    flat_features = []

    for i in range(len(contents)):
        flat_contents += contents[i]

    for i in range(len(features)):
        flat_features += features[i]

    # obtain a tokenizer
    print("\nmaximum words to work with: ", max_words_length)
    t_content = Tokenizer(num_words=max_words_length, filters='')
    t_features =  Tokenizer(num_words=max_words_length, filters='')
    t_content.fit_on_texts(flat_contents)
    t_features.fit_on_texts(flat_features)

    print("\nbuilding contents and features dict")
    if (max_words_length is not None):
        vals_content = t_content.word_index.items()
        vals_content = sorted(vals_content, key=lambda x: x[1])
        vals_feature = t_features.word_index.items()
        vals_feature = sorted(vals_feature, key=lambda x: x[1])

        with open('original_data/word_vocab.txt', 'w') as f:
            for key, value in vals_content[:max_words_length]:
                f.write('{0} {1}'.format(key, value))
                f.write('\n')

        with open('original_data/feature_vocab.txt', 'w') as f:
            for key, value in vals_feature[:max_words_length]:
                f.write('{0} {1}'.format(key, value))
                f.write('\n')

    else:
        with open('original_data/word_vocab.txt', 'w') as f:
            for key, value in t_content.word_index.items():
                f.write('{0} {1}'.format(key, value))
                f.write('\n')

        with open('original_data/feature_vocab.txt', 'w') as f:
            for key, value in t_features.word_index.items():
                f.write('{0} {1}'.format(key, value))
                f.write('\n')


if __name__ == '__main__':
    df, df_train, df_valid, df_test = data_cleanse('original_data/original.csv')
    df.to_csv('original_data/preprocessed.csv')
    df_train.to_csv('original_data/train.csv')
    df_valid.to_csv('original_data/valid.csv')
    df_test.to_csv('original_data/test.csv')

    create_table_summary('original_data/train.csv', 'train_table.txt', 'train_summary.txt')
    create_table_summary('original_data/valid.csv', 'valid_table.txt', 'valid_summary.txt')
    create_table_summary('original_data/test.csv', 'test_table.txt', 'test_summary.txt')

    create_vocab('original_data/preprocessed.csv', 10000)


