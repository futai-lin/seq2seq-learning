#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-17 下午8:36
# @Author  : Zhuoyu


import tensorflow as tf
import pickle


class OutputCell(object):
    def __init__(self, inp_size, op_size, scope_name):
        self.inp_size = inp_size
        self.op_size = op_size
        self.scope_name = scope_name
        self.params = {}

        with tf.variable_scope(scope_name):
            self.W = tf.get_variable('W', [inp_size, op_size])
            self.b = tf.get_variable('b', [op_size], initializer=tf.zeros_initializer(), dtype=tf.float32)

        self.params.update({'W': self.W, 'b': self.b})

    def __call__(self, x, finished = None):
        out = tf.nn.xw_plus_b(x, self.W, self.b)

        if finished is not None:
            out = tf.where(finished, tf.zeros_like(out), out)
            #out = tf.multiply(1 - finished, out)
        return out

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
