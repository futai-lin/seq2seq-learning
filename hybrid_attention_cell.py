#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-17 下午2:41
# @Author  : Zhuoyu

import tensorflow as tf
import pickle


class HybridAttentionWrapper(object):
    def __init__(self, hid_size, inp_size, fe_size, en_val_s, en_fea_s, scope_name):
        '''
        @param: hid_size: the size of lstm cell
        @param: inpu_size: the size of input
        @param: fe_size: the size of feature
        @param: en_value_s: the state of value encoder
        @param: en_fea_s: the state of feature encoder
        '''
        
        self.en_val_s = tf.transpose(en_val_s, [1,0,2])  # input_len * batch * input_size
        self.en_fea_s = tf.transpose(en_fea_s, [1,0,2])
        self.hid_size = hid_size
        self.inp_size = inp_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W_en_hid = tf.get_variable('W_en_hid', [inp_size, hid_size])
            self.b_en_hid = tf.get_variable('b_en_hid', [hid_size])
            self.W_de_hid = tf.get_variable('W_de_hid', [inp_size, hid_size])
            self.b_de_hid = tf.get_variable('b_de_hid', [hid_size])
            self.W_out = tf.get_variable('W_out', [2*inp_size, hid_size])
            self.b_out = tf.get_variable('b_out', [hid_size])
            self.W_fe = tf.get_variable('W_fe', [fe_size, hid_size])
            self.b_fe = tf.get_variable('b_fe', [hid_size])
            self.W_c = tf.get_variable('W_c', [inp_size, hid_size])
            self.b_c = tf.get_variable('b_c', [hid_size])

        self.params.update({'W_en_hid': self.W_en_hid, 'W_de_hid': self.W_de_hid, 'W_out': self.W_out,
                            'b_en_hid': self.b_en_hid, 'b_de_hid': self.b_de_hid, 'b_out': self.b_out,
                            'W_fe': self.W_fe, 'W_c': self.W_c, 
                            'b_fe': self.b_fe, 'b_c': self.b_c})

        en_val_s_2d = tf.reshape(self.en_val_s, [-1, inp_size])
        phi_en_val_2d = tf.tanh(tf.nn.xw_plus_b(en_val_s_2d, self.W_en_hid, self.b_en_hid))
        self.phi_en_val = tf.reshape(phi_en_val_2d, tf.shape(self.en_val_s))
        en_fea_s_2d = tf.reshape(self.en_fea_s, [-1, fe_size])
        phi_en_fea_2d = tf.tanh(tf.nn.xw_plus_b(en_fea_s_2d, self.W_fe, self.b_fe))
        self.phi_en_fea = tf.reshape(phi_en_fea_2d, tf.shape(self.en_val_s))

    def __call__(self, x, coverage = None, finished = None):
        gamma_de = tf.tanh(tf.nn.xw_plus_b(x, self.W_de_hid, self.b_de_hid))  # batch * hidden_size
        alpha_de = tf.tanh(tf.nn.xw_plus_b(x, self.W_c, self.b_c))
        fea_weights = tf.reduce_sum(self.phi_en_fea * alpha_de, reduction_indices=2, keep_dims=True)
        fea_weights = tf.exp(fea_weights - tf.reduce_max(fea_weights, reduction_indices=0, keep_dims=True))
        fea_weights = tf.divide(fea_weights, (1e-6 + tf.reduce_sum(fea_weights, reduction_indices=0, keep_dims=True)))
        
        
        att_weights = tf.reduce_sum(self.phi_en_val * gamma_de, reduction_indices=2, keep_dims=True)  # input_len * batch
        att_weights = tf.exp(att_weights - tf.reduce_max(att_weights, reduction_indices=0, keep_dims=True))
        att_weights = tf.divide(att_weights, (1e-6 + tf.reduce_sum(att_weights, reduction_indices=0, keep_dims=True)))
        att_weights = tf.divide(att_weights * fea_weights, (1e-6 + tf.reduce_sum(att_weights * fea_weights, reduction_indices=0, keep_dims=True)))
        
        context = tf.reduce_sum(self.en_val_s * att_weights, reduction_indices=0)  # batch * input_size
        out = tf.tanh(tf.nn.xw_plus_b(tf.concat([context, x], -1), self.W_out, self.b_out))

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
        return out, att_weights

    def save(self, path):
        param_vals = {}
        for param in self.params:
            param_vals[param] = self.params[param].eval()
        with open(path, 'wb') as f:
            pickle.dump(param_vals, f, True)

    def load(self, path):
        param_vals = pickle.load(open(path, 'rb'))
        for param in param_vals:
            self.params[param].load(param_vals[param])
