#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-13 下午2:40
# @Author  : Futai Lin

import tensorflow as tf
import time
import numpy as np


class DataLoader(object):
    def __init__(self, data_dir, limits):
        self.train_data_path = [data_dir + '/train/train_summary_id', data_dir + '/train/train_table_val_id',
                                data_dir + '/train/train_table_lab_id', data_dir + '/train/train_table_lpos',
                                data_dir + '/train/train_table_rpos']
        self.test_data_path = [data_dir + '/test/test_summary_id', data_dir + '/test/test_table_val_id',
                               data_dir + '/test/test_table_lab_id', data_dir + '/test/test_table_lpos',
                               data_dir + '/test/test_table_rpos']
        self.dev_data_path = [data_dir + '/valid/valid_summary_id', data_dir + '/valid/valid_table_val_id',
                              data_dir + '/valid/valid_table_lab_id', data_dir + '/valid/valid_table_lpos',
                              data_dir + '/valid/valid_table_rpos']
        self.limits = limits
        self.man_text_len = 100
        start_time = time.time()

        print('Reading datasets ...')
        self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)
        # self.small_test_set = self.load_data(self.small_test_data_path)
        self.dev_set = self.load_data(self.dev_data_path)
        print ('Reading datasets consumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        summary_path, text_path, field_path, pos_path, rpos_path = path
        summaries = open(summary_path, 'r').read().strip().split('\n')
        texts = open(text_path, 'r').read().strip().split('\n')
        fields = open(field_path, 'r').read().strip().split('\n')
        poses = open(pos_path, 'r').read().strip().split('\n')
        rposes = open(rpos_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            summaries = summaries[:self.limits]
            texts = texts[:self.limits]
            fields = fields[:self.limits]
            poses = poses[:self.limits]
            rposes = rposes[:self.limits]
        print(summaries[0].strip().split(' '))
        summaries = [list(map(int, summary.strip().split(' '))) for summary in summaries]
        texts = [list(map(int, text.strip().split(' '))) for text in texts]
        fields = [list(map(int, field.strip().split(' '))) for field in fields]
        poses = [list(map(int, pos.strip().split(' '))) for pos in poses]
        rposes = [list(map(int, rpos.strip().split(' '))) for rpos in rposes]
        return summaries, texts, fields, poses, rposes

    def batch_iter(self, data, batch_size, shuffle):
        summaries, texts, fields, poses, rposes = data
        data_size = len(summaries)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            summaries = np.array(summaries)[shuffle_indices]
            texts = np.array(texts)[shuffle_indices]
            fields = np.array(fields)[shuffle_indices]
            poses = np.array(poses)[shuffle_indices]
            rposes = np.array(rposes)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_summary_len = max([len(sample) for sample in summaries[start_index:end_index]])
            max_text_len = max([len(sample) for sample in texts[start_index:end_index]])
            batch_data = {'en_input':[], 'en_fe':[], 'en_lpos':[], 'en_rpos':[], 'en_len':[],
                          'de_in':[], 'de_len':[], 'de_output':[]}

            for summary, text, field, pos, rpos in zip(summaries[start_index:end_index], texts[start_index:end_index],
                                            fields[start_index:end_index], poses[start_index:end_index],
                                            rposes[start_index:end_index]):
                summary_len = len(summary)
                text_len = len(text)
                pos_len = len(pos)
                rpos_len = len(rpos)
                assert text_len == len(field)
                assert pos_len == len(field)
                assert rpos_len == pos_len
                gold = summary + [2] + [0] * (max_summary_len - summary_len)
                summary = summary + [0] * (max_summary_len - summary_len)
                text = text + [0] * (max_text_len - text_len)
                field = field + [0] * (max_text_len - text_len)
                pos = pos + [0] * (max_text_len - text_len)
                rpos = rpos + [0] * (max_text_len - text_len)
                
                if max_text_len > self.man_text_len:
                    text = text[:self.man_text_len]
                    field = field[:self.man_text_len]
                    pos = pos[:self.man_text_len]
                    rpos = rpos[:self.man_text_len]
                    text_len = min(text_len, self.man_text_len)
                
                batch_data['en_input'].append(text)
                batch_data['en_len'].append(text_len)
                batch_data['en_fe'].append(field)
                batch_data['en_lpos'].append(pos)
                batch_data['en_rpos'].append(rpos)
                batch_data['de_in'].append(summary)
                batch_data['de_len'].append(summary_len)
                batch_data['de_output'].append(gold)
  
            yield batch_data