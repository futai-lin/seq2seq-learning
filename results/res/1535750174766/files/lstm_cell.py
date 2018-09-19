#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-17 下午8:34
# @Author  : Zhuoyu

import tensorflow as tf
import pickle


class LstmCell(object):
    def __init__(self, hid_size, inp_size, scope_name):
        '''
        :param hidden_size: the size of hidden layer
        :param input_size: the size of input layer
        :param scope_name: the name of variable_scope
        '''
        self.hid_size = hid_size
        self.inp_size = inp_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [self.inp_size + self.hid_size, 4 * self.hid_size])
            self.b = tf.get_variable('b', [4 * self.hid_size], dtype=tf.float32)

        self.params.update({'W':self.W, 'b':self.b})

    def __call__(self, x, s_t, finished = None):
        h_prev, c_prev = s_t

        x = tf.concat([x, h_prev], 1)
        i, j, f, o = tf.split(tf.nn.xw_plus_b(x, self.W, self.b), 4, 1)

        # Final Memory cell
        c = tf.sigmoid(f+1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j)
        h = tf.sigmoid(o) * tf.tanh(c)

        out, state = h, (h, c)
        if finished is not None:
            out = tf.where(finished, tf.zeros_like(h), h)
            state = (tf.where(finished, h_prev, h), tf.where(finished, c_prev, c))
            # out = tf.multiply(1 - finished, h)
            # state = (tf.multiply(1 - finished, h) + tf.multiply(finished, h_prev),
            #          tf.multiply(1 - finished, c) + tf.multiply(finished, c_prev))

        return out, state

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