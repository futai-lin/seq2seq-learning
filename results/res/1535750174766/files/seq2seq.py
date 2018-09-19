#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-18 下午8:37
# @Author  : Futai

import tensorflow as tf
import pickle
from attention_cell import AttentionWrapper
from hybrid_attention_cell import HybridAttentionWrapper
from lstm_cell import LstmCell
from fgate_lstm_cell import FgateLstmCell
from output_cell import OutputCell


class Seq2Seq(object):
    def __init__(self, batch_size, hid_size, emb_size, fe_size, pos_size, source_vocab, fe_vocab,
                 pos_vocab, target_vocab, fe_concat, pos_concat, fgate_en, hybrid_att,
                 en_add_pos, de_add_pos, learning_rate, scope_name, name, start_token=2, stop_token=2, max_length=150):
        '''
        @param batch_size: size of batch
        @param hid_size, emb_size, fe_size, pos_size: size of hidden layer
        @param source_vocab, fe_vocab, pos_vocab: size of encoder vocabulary
        @param target_vocab: size of decoder vocabulary 
        @param fe_concat, pos_concat: whether concat feature, position embedding or not
        @param fgate_en, hybrid_att: whether use feature-gate, hybrid attention or not
        @param en_add_pos, de_add_pos: whether add position embedding to feature-gate encoder, decoder with hybrid attention or not
        '''
        self.batch_size = batch_size
        self.hid_size = hid_size
        self.emb_size = emb_size
        self.fe_size = fe_size
        self.pos_size = pos_size
        self.unified_size = emb_size if not fe_concat else emb_size + fe_size
        self.unified_size = self.unified_size if not pos_concat else self.unified_size + 2 * pos_size
        self.fe_en_size = fe_size if not en_add_pos else fe_size + 2 * pos_size
        self.fe_att_size = fe_size if not de_add_pos else fe_size + 2 * pos_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.fe_vocab = fe_vocab
        self.pos_vocab = pos_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.name = name
        self.fe_concat = fe_concat
        self.pos_concat = pos_concat
        self.fgate_en = fgate_en
        self.hybrid_att = hybrid_att
        self.en_add_pos = en_add_pos
        self.de_add_pos = de_add_pos

        self.units = {}
        self.params = {}

        self.en_input = tf.placeholder(tf.int32, [None, None])
        self.en_fe = tf.placeholder(tf.int32, [None, None])
        self.en_lpos = tf.placeholder(tf.int32, [None, None])
        self.en_rpos = tf.placeholder(tf.int32, [None, None])
        self.de_input = tf.placeholder(tf.int32, [None, None])
        self.en_len = tf.placeholder(tf.int32, [None])
        self.de_len = tf.placeholder(tf.int32, [None])
        self.de_output = tf.placeholder(tf.int32, [None, None])
        self.en_mask = tf.sign(tf.to_float(self.en_lpos))
        with tf.variable_scope(scope_name):
            if self.fgate_en:
                print('feature-gate encoder LSTM')
                self.en_lstm = FgateLstmCell(self.hid_size, self.unified_size, self.fe_en_size, 'encoder_select')
            else:
                print('regular encoder LSTM')
                self.en_lstm = LstmCell(self.hid_size, self.unified_size, 'encoder_lstm')
            self.de_lstm = LstmCell(self.hid_size, self.emb_size, 'decoder_lstm')
            self.de_out = OutputCell(self.hid_size, self.target_vocab, 'decoder_output')

        self.units.update({'encoder_lstm': self.en_lstm,'decoder_lstm': self.de_lstm,
                           'decoder_output': self.de_out})

        # ======================================== embeddings ======================================== #
        with tf.device('/cpu:0'):
            with tf.variable_scope(scope_name):
                self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
                self.en_embed = tf.nn.embedding_lookup(self.embedding, self.en_input)
                self.de_embed = tf.nn.embedding_lookup(self.embedding, self.de_input)
                if self.fe_concat or self.fgate_en or self.en_add_pos or self.de_add_pos:
                    self.fembedding = tf.get_variable('fembedding', [self.fe_vocab, self.fe_size])
                    self.fe_embed = tf.nn.embedding_lookup(self.fembedding, self.en_fe)
                    self.fe_pos_embed = self.fe_embed
                    if self.fe_concat:
                        self.en_embed = tf.concat([self.en_embed, self.fe_embed], 2)
                if self.pos_concat or self.en_add_pos or self.de_add_pos:
                    self.lembedding = tf.get_variable('lembedding', [self.pos_vocab, self.pos_size])
                    self.rembedding = tf.get_variable('rembedding', [self.pos_vocab, self.pos_size])
                    self.lpos_embed = tf.nn.embedding_lookup(self.lembedding, self.en_lpos)
                    self.rpos_embed = tf.nn.embedding_lookup(self.rembedding, self.en_rpos)
                    if pos_concat:
                        self.en_embed = tf.concat([self.en_embed, self.lpos_embed, self.rpos_embed], 2)
                        self.fe_pos_embed = tf.concat([self.fe_embed, self.lpos_embed, self.rpos_embed], 2)
                    elif self.en_add_pos or self.de_add_pos:
                        self.fe_pos_embed = tf.concat([self.fe_embed, self.lpos_embed, self.rpos_embed], 2)

        if self.fe_concat or self.fgate_en:
            self.params.update({'fembedding': self.fembedding})
        if self.pos_concat or self.en_add_pos or self.de_add_pos:
            self.params.update({'lembedding': self.lembedding})
            self.params.update({'rembedding': self.rembedding})
        self.params.update({'embedding': self.embedding})

        # ======================================== encoder ======================================== #
        if self.fgate_en:
            print('feature-gate encoder used')
            en_outputs, en_state = self.fgate_encoder(self.en_embed, self.fe_pos_embed, self.en_len)
        else:
            print('regular encoder used')
            en_outputs, en_state = self.encoder(self.en_embed, self.en_len)
        # ======================================== decoder ======================================== #

        if self.hybrid_att:
            print('hybrid attention mechanism used')
            with tf.variable_scope(scope_name):
                self.att_layer = HybridAttentionWrapper(self.hid_size, self.hid_size, self.fe_att_size,
                                                        en_outputs, self.fe_pos_embed, "attention")
                self.units.update({'attention': self.att_layer})
        else:
            print("regular attention used")
            with tf.variable_scope(scope_name):
                self.att_layer = AttentionWrapper(self.hid_size, self.hid_size, en_outputs, "attention")
                self.units.update({'attention': self.att_layer})


        # decoder for training
        de_outputs, de_state = self.decoder_train(en_state, self.de_embed, self.de_len)
        # decoder for testing
        self.g_tokens, self.atts = self.decoder_test(en_state)
        # self.beam_seqs, self.beam_probs, self.cand_seqs, self.cand_probs = self.decoder_beam(en_state, beam_size)
        

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs, labels=self.de_output)
        mask = tf.sign(tf.to_float(self.de_output))
        losses = mask * losses
        self.mean_loss = tf.reduce_mean(losses)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.en_input)[0]
        max_time = tf.shape(self.en_input)[1]
        hid_size = self.hid_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hid_size], dtype=tf.float32),
              tf.zeros([batch_size, hid_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.en_lstm(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.unified_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def fgate_encoder(self, inputs, features, inputs_len):
        batch_size = tf.shape(self.en_input)[0]
        max_time = tf.shape(self.en_input)[1]
        hid_size = self.hid_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hid_size], dtype=tf.float32),
              tf.zeros([batch_size, hid_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        fe_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        fe_ta = fe_ta.unstack(tf.transpose(features, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, f_t, s_t, emit_ta, finished):
            o_t, s_nt = self.en_lstm(x_t, f_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.unified_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            f_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.fe_att_size], dtype=tf.float32),
                                     lambda: fe_ta.read(t+1))
            return t+1, x_nt, f_nt, s_nt, emit_ta, finished

        _, _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), fe_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state


    def decoder_train(self, init_state, inputs, inputs_len):
        batch_size = tf.shape(self.de_input)[0]
        max_time = tf.shape(self.de_input)[1]
        encoder_len = tf.shape(self.en_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = init_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.de_lstm(x_t, s_t, finished)
            o_t, _ = self.att_layer(o_t)
            o_t = self.de_out(o_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta,  _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def decoder_test(self, init_state):
        batch_size = tf.shape(self.en_input)[0]
        encoder_len = tf.shape(self.en_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = init_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        att_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, att_ta, finished):
            o_t, s_nt = self.de_lstm(x_t, s_t, finished)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.de_out(o_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            att_ta = att_ta.write(t, w_t)
            next_token = tf.arg_max(o_t, 1)
            x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, s_nt, emit_ta, att_ta, finished

        _, _, state, emit_ta, att_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, att_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.arg_max(outputs, 2)
        atts = att_ta.stack()
        return pred_tokens, atts


    def decoder_beam(self, init_state, beam_size):

        def beam_init():
            # return b_s_1 b_p_1 c_s_1 c_p_1 next_states time
            time_1 = tf.constant(1, dtype=tf.int32)
            b_s_0 = tf.constant([[self.start_token]]*beam_size)
            b_p_0 = tf.constant([0.]*beam_size)

            c_s_0 = tf.constant([[self.start_token]])
            c_p_0 = tf.constant([-3e38])

            b_s_0._shape = tf.TensorShape((None, None))
            b_p_0._shape = tf.TensorShape((None,))
            c_s_0._shape = tf.TensorShape((None, None))
            c_p_0._shape = tf.TensorShape((None,))
            
            inps = [self.start_token]
            x_t = tf.nn.embedding_lookup(self.embedding, inps)
            print(x_t.get_shape().as_list())
            o_t, s_nt = self.de_lstm(x_t, init_state)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.de_out(o_t)
            print(s_nt[0].get_shape().as_list())
            
            log_prob_2d = tf.nn.log_softmax(o_t)
            t_p = log_prob_2d + tf.reshape(b_p_0, [-1, 1])
            t_p_no_EOS = tf.concat([tf.slice(t_p, [0, 0], [1, self.stop_token]),
                               tf.tile([[-3e38]], [1, 1]),
                               tf.slice(t_p, [0, self.stop_token + 1],
                                        [1, self.target_vocab - self.stop_token - 1])], 1)
            flat_t_p = tf.reshape(t_p_no_EOS, [-1])
            print(flat_t_p.get_shape().as_list())

            b_k = tf.minimum(tf.size(flat_t_p), beam_size)
            next_b_p, top_idx = tf.nn.top_k(flat_t_p, k=b_k)

            next_base = tf.floordiv(top_idx, self.target_vocab)
            next_mod = tf.mod(top_idx, self.target_vocab)

            next_b_s = tf.concat([tf.gather(b_s_0, next_base),
                                        tf.reshape(next_mod, [-1, 1])], 1)

            c_s_p = tf.pad(c_s_0, [[0, 0], [0, 1]])
            b_s_EOS = tf.pad(b_s_0, [[0, 0], [0, 1]])
            new_c_s = tf.concat([c_s_p, b_s_EOS], 0)
            print(new_c_s.get_shape().as_list())

            EOS_p = tf.slice(t_p, [0, self.stop_token], [beam_size, 1])
            new_c_p = tf.concat([c_p_0, tf.reshape(EOS_p, [-1])], 0)
            c_k = tf.minimum(tf.size(new_c_p), self.beam_size)
            next_c_p, next_c_indx = tf.nn.top_k(new_c_p, k=c_k)
            next_c_s = tf.gather(new_c_s, next_c_indx)

            part_sta_0 = tf.reshape(tf.stack([s_nt[0]] * beam_size), [beam_size, self.hid_size])
            part_sta_1 = tf.reshape(tf.stack([s_nt[1]] * beam_size), [beam_size, self.hid_size])
            part_sta_0._shape = tf.TensorShape((None, None))
            part_sta_1._shape = tf.TensorShape((None, None))
            next_sta = (part_sta_0, part_sta_1)
            print(next_sta[0].get_shape().as_list())
            return next_b_s, next_b_p, next_c_s, next_c_p, next_sta, time_1

        b_s_1, b_p_1, c_s_1, c_p_1, sta_1, time_1 = beam_init()
        b_s_1._shape = tf.TensorShape((None, None))
        b_p_1._shape = tf.TensorShape((None,))
        c_s_1._shape = tf.TensorShape((None, None))
        c_p_1._shape = tf.TensorShape((None,))
        
        def beam_step(b_s, b_p, c_s, c_p, sta, time):
            '''
            @param b_s : [beam_size, time]
            @param b_p: [beam_size, ]
            @param c_s : [beam_size, time]
            @param c_p: [beam_size, ]
            @param sta : [beam_size * hidden_size, beam_size * hidden_size]
            '''
            inps = tf.reshape(tf.slice(b_s, [0, time], [beam_size, 1]), [beam_size])
            # print inputs.get_shape().as_list()
            x_t = tf.nn.embedding_lookup(self.embedding, inps)
            # print(x_t.get_shape().as_list())
            o_t, s_nt = self.de_lstm(x_t, states)
            o_t, w_t = self.att_layer(o_t)
            o_t = self.de_out(o_t)
            log_prob_2d = tf.nn.log_softmax(o_t)
            print(log_prob_2d.get_shape().as_list())
            t_p = log_prob_2d + tf.reshape(b_p, [-1, 1])
            print(t_p.get_shape().as_list())
            t_p_no_EOS = tf.concat([tf.slice(t_p, [0, 0], [beam_size, self.stop_token]),
                                           tf.tile([[-3e38]], [beam_size, 1]),
                                           tf.slice(t_p, [0, self.stop_token + 1],
                                                    [beam_size, self.target_vocab - self.stop_token - 1])], 1)
            print(t_p_no_EOS.get_shape().as_list())
            flat_t_p = tf.reshape(t_p_no_EOS, [-1])
            print(flat_t_p.get_shape().as_list())

            b_k = tf.minimum(tf.size(flat_t_p), beam_size)
            next_b_p, top_indx = tf.nn.top_k(flat_total_probs, k=b_k)
            print(next_b_p.get_shape().as_list())

            next_base = tf.floordiv(top_indx, self.target_vocab)
            next_mod = tf.mod(top_indx, self.target_vocab)
            print(next_mod.get_shape().as_list())

            next_b_s = tf.concat([tf.gather(b_s, next_base),
                                        tf.reshape(next_mod, [-1, 1])], 1)
            next_sta = (tf.gather(s_nt[0], next_base), tf.gather(s_nt[1], next_base))
            print(next_b_s.get_shape().as_list())

            c_s_p = tf.pad(c_s, [[0, 0], [0, 1]])
            b_s_EOS = tf.pad(b_s, [[0, 0], [0, 1]])
            new_c_s = tf.concat([c_s_p, b_s_EOS], 0) 
            print(new_c_s.get_shape().as_list())

            EOS_prob = tf.slice(t_p, [0, self.stop_token], [beam_size, 1])
            new_c_p = tf.concat([c_p, tf.reshape(EOS_prob, [-1])], 0)
            c_k = tf.minimum(tf.size(new_c_p), self.beam_size)
            next_c_p, next_c_indx = tf.nn.top_k(new_c_p, k=c_k)
            next_c_s = tf.gather(new_c_s, next_c_indx)

            return next_b_s, next_b_p, next_c_s, next_c_p, next_sta, time+1
        
        def beam_cond(b_p, b_s, c_p, c_s, sta, time):
            length =  (tf.reduce_max(b_p) >= tf.reduce_min(c_p))
            return tf.logical_and(length, tf.less(time, 60))

        loop_var = [b_s_1, b_p_1, c_s_1, c_p_1, sta_1, time_1]
        ret_var = tf.while_loop(cond=beam_cond, body=beam_step, loop_vars=loop_var, back_prop=False)
        b_s_all, b_p_all, c_s_all, c_p_all, _, time_all = ret_var

        return b_s_all, b_p_all, c_s_all, c_p_all

    def __call__(self, x, sess):
        loss,  _ = sess.run([self.mean_loss, self.train_op],
                           {self.en_input: x['en_input'], self.en_len: x['en_len'], 
                            self.en_fe: x['en_fe'], self.en_lpos: x['en_lpos'], 
                            self.en_rpos: x['en_rpos'], self.de_input: x['de_in'],
                            self.de_len: x['de_len'], self.de_output: x['de_output']})
        return loss

    def generate(self, x, sess):
        preds, atts = sess.run([self.g_tokens, self.atts],
                               {self.en_input: x['en_input'], self.en_fe: x['en_fe'], 
                                self.en_len: x['en_len'], self.en_lpos: x['en_lpos'],
                                self.en_rpos: x['en_rpos']})
        return preds, atts

    def generate_beam(self, x, sess):
        # beam_seqs_all, beam_probs_all, cand_seqs_all, cand_probs_all
        b_s_all, b_p_all, c_s_all, c_p_all = sess.run(
                         [self.b_s,self.b_p, self.c_s, self.c_p],
                         {self.en_input: x['en_input'], self.en_fe: x['en_fe'],
                          self.en_len: x['en_len'], self.en_lpos: x['en_lpos'],
                          self.en_rpos: x['enc_rpos']})
        return b_s_all, b_p_all, c_s_all, c_p_all

    def save(self, path):
        for u in self.units:
            self.units[u].save(path+u+".pkl")
        param_vals = {}
        for param in self.params:
            param_vals[param] = self.params[param].eval()
        with open(path+self.name+".pkl", 'wb') as f:
            pickle.dump(param_vals, f, True)

    def load(self, path):
        for u in self.units:
            self.units[u].load(path+u+".pkl")
        param_vals = pickle.load(open(path+self.name+".pkl", 'rb'))
        for param in param_vals:
            self.params[param].load(param_vals[param])
