#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-17 下午10:16
# @Author  : Zhuoyu

import tensorflow as tf
import pickle


class FgateLstmCell(object):
    def __init__(self, hid_size, inp_size, fe_size, scope_name):
        '''
        :param hidden_size: the size of hidden layer
        :param input_size: the size of input layer(word)
        :param field_size: the size of input layer(field)
        :param scope_name: the name of variable_scope
        '''
        self.hid_size = hid_size
        self.inp_size = inp_size
        self.fe_size = fe_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W1 = tf.get_variable('W1', [self.inp_size + self.hid_size, 4*self.hid_size])
            self.b1 = tf.get_variable('b1', [4 * self.hid_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
            self.W2 = tf.get_variable('W2', [self.fe_size, 2 * self.hid_size])
            self.b2 = tf.get_variable('b2', [2 * hid_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
        self.params.update({'W1':self.W1, 'b1':self.b1, 'W2':self.W2, 'b2':self.b2})

    def __call__(self, x, fe, s_t, finished = None):
        """
        :param x: batch * input
        :param s: (h,s,d)
        :param finished:
        :return:
        """
        h_prev, c_prev = s_t  # batch * hidden_size

        x = tf.concat([x, h_prev], 1)
        # fd = tf.concat([fd, h_prev], 1)
        i, j, f, o = tf.split(tf.nn.xw_plus_b(x, self.W1, self.b1), 4, 1)
        r, d = tf.split(tf.nn.xw_plus_b(fe, self.W2, self.b2), 2, 1)
        # Final Memory cell
        c = tf.sigmoid(f+1.0) * c_prev + tf.sigmoid(i) * tf.tanh(j) + tf.sigmoid(r) * tf.tanh(d)  # batch * hidden_size
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