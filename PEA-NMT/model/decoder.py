import theano
import theano.tensor as T

import nn
import ops
from bridge import *


class Decoder:
    def __init__(self, dim_y, dim_hid, dim_key, dim_value, n_y_vocab, *args, **kwargs):
        """
        :param dim_y: 
        :param dim_hid: dimension of decoder's hidden state 
        :param dim_key: dimension of query keys
        :param dim_value: dimension of context values
        """
        self.dim_y = dim_y
        self.dim_hid = dim_hid
        self.dim_key = dim_key
        self.dim_value = dim_value

    def step(self, y_prev, mask, state, cwkeys, cpkeys, fwkeys, fpkeys, values, key_mask):
        """
        forward step
        return (state, context) pair
        """
        raise NotImplementedError

    def prediction(self, y_emb, state, context, y_pos, keep_prob=1.0):
        raise NotImplementedError

    def build_sampling(self, src_seq, src_mask, target_embedding, target_bias, keys, values, initial_state):
        # sampling graph, this feature is optional
        raise NotImplementedError

    def build_attention(self, src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, keys, values,
                        initial_state):
        # attention graph, this feature is optional
        raise NotImplementedError

    def get_cost(self, y_seq, mask, probs):
        assert probs.ndim == 2
        idx = T.arange(y_seq.flatten().shape[0])
        ce = -T.log(probs[idx, y_seq.flatten()])
        ce = ce.reshape(y_seq.shape)
        ce = T.sum(ce * mask, 0)

        cost = T.mean(ce)

        snt_cost = ce
        return cost, snt_cost

    def scan(self, y_emb, mask, cwkeys, cpkeys, fwkeys, fpkeys, key_mask, values, initial_state):
        """
        build model
        :return: 
        """
        seq = [y_emb, mask]
        outputs_info = [initial_state, None, None, None]
        non_seq = [cwkeys, cpkeys, fwkeys, fpkeys, values, key_mask]
        (states, contexts, y_pos_states, posscores) = ops.scan(self.step, seq, outputs_info, non_seq)

        return states, contexts, y_pos_states, posscores

    def forward(self, y_seq, y_emb, mask, cwkeys, cpkeys, fwkeys, fpkeys, key_mask, values, initial_state, pos_seq, keep_prob=1.0):
        """
        return states,contexts,cost 
        """
        raise NotImplementedError


class DecoderGruCond(Decoder):
    """
    prediction: s1, y0 -> y1 
    recurrence: s0, y0 -> s1
    """

    def __init__(self, dim_y, dim_hid, dim_key, dim_value, dim_readout, n_y_vocab, *args, **kwargs):
        """
        see `https://github.com/nyu-dl/dl4mt-tutorial/blob/master/docs/cgru.pdf`
        1. s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        2. c_j = att(H, s_j^{\prime})
        3. s_j = GRU^2(c_j, s_j^{\prime})
        """
        Decoder.__init__(self, dim_y, dim_hid, dim_key, dim_value, n_y_vocab)
        self.dim_readout = dim_readout
        self.n_y_vocab = n_y_vocab

        self.n_y_tagvocab = kwargs["n_y_tagvocab"]
        self.poshdim = kwargs["poshdim"]
        self.posnndim = kwargs["posnndim"]

        self.dim_word2pos = kwargs["word2pos"]
        self.dim_pos2word = kwargs["pos2word"]
        self.dim_pos2pos = kwargs["pos2pos"]

        # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        self.cell1 = nn.rnn_cell.gru_cell([dim_y, dim_hid])
        self.dim_query = dim_hid

        self.dim_posquery = self.posnndim + self.poshdim

        # s_j = GRU^2(c_j, s_j^{\prime})
        self.cell2 = nn.rnn_cell.gru_cell([dim_value, dim_hid])

    def step(self, y_prev, mask, state, src_words_keys, src_pos_keys, pos_words_keys, pos_pos_keys, values, key_mask):
        mask = mask[:, None]

        # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
        _, state_prime = self.cell1(y_prev, state, scope="gru1")
        state_prime = (1.0 - mask) * state + mask * state_prime
        # c_j = att(H, s_j^{\prime})
        alpha = coarseattention(state_prime, [src_words_keys, src_pos_keys], key_mask, self.dim_query, [self.dim_key, self.dim_word2pos])
        context = T.sum(alpha[:, :, None] * values, 0)
        # s_j = GRU^2(c_j, s_j^{\prime})
        output, next_state = self.cell2(context, state_prime, scope="gru2")
        next_state = (1.0 - mask) * state + mask * next_state

        # y_pos_{j} = nn(s_{j})
        tempstate = nn.feedforward(next_state,
                                   [self.dim_hid, self.posnndim],
                                          True, activation=T.nnet.relu, scope="ttaggerstates")

        score = nn.linear(tempstate, [self.posnndim, self.n_y_tagvocab], True, scope="ttaggerscores")
        prob = T.nnet.softmax(score)

        with ops.variable_scope("tgttag_embedding"):
            tgttag_embedding = ops.get_variable("embedding", [self.n_y_tagvocab, self.poshdim])
            tgttag_bias = ops.get_variable("bias", [self.poshdim])

        y_pos_state = T.dot(prob, tgttag_embedding) + tgttag_bias

        posquery = T.concatenate([tempstate, y_pos_state], -1)
        beta = fineattention([state_prime, posquery], [[src_words_keys, src_pos_keys], [pos_words_keys, pos_pos_keys]], key_mask,
                             [self.dim_query, self.dim_posquery], [[self.dim_key, self.dim_word2pos], [self.dim_pos2word, self.dim_pos2pos]])

        finalalpha = beta
        # adl = ops.get_variable("adaptive", [])
        # walpha = T.exp(adl)
        # wbeta = T.exp(1.0 - adl)
        # finalalpha = walpha/(walpha+wbeta) * alpha + wbeta/(walpha+wbeta) * beta
        #
        # finalalpha = 0.5 * beta + 0.5 * alpha
        context = T.sum(finalalpha[:, :, None] * values, 0)

        return next_state, context, y_pos_state, prob

    def build_sampling(self, src_seq, src_mask, target_embedding, target_bias, keys, values, initial_state):
        # sampling graph, this feature is optional
        max_len = T.iscalar()

        def sampling_loop(inputs, state, keys, values, key_mask):
            _, state_prime = self.cell1(inputs, state, scope="gru1")
            alpha = attention(state_prime, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            output, next_state = self.cell2(context, state_prime, scope="gru2")
            probs = self.prediction(inputs, next_state, context)  # p(y_j) \propto f(y_{j-1}, c_j, s_j)
            next_words = ops.random.multinomial(probs).argmax(axis=1)
            new_inputs = nn.embedding_lookup(target_embedding, next_words)
            new_inputs = new_inputs + target_bias

            return [next_words, new_inputs, next_state]

        with ops.variable_scope("decoder"):
            batch = src_seq.shape[1]
            initial_inputs = T.zeros([batch, self.dim_y], theano.config.floatX)

            outputs_info = [None, initial_inputs, initial_state]
            nonseq = [keys, values, src_mask]
            outputs, updates = theano.scan(sampling_loop, [], outputs_info,
                                           nonseq, n_steps=max_len)
            sampled_words = outputs[0]

        sampling_inputs = [src_seq, src_mask, max_len]
        sampling_outputs = sampled_words
        sample = theano.function(sampling_inputs, sampling_outputs,
                                 updates=updates)
        return sample

    def build_attention(self, src_seq, src_mask, target_inputs, tgt_seq, tgt_mask, keys, values, initial_state):
        # attention graph, this feature is optional
        def attention_loop(inputs, mask, state, keys, values, key_mask):
            mask = mask[:, None]
            # s_j^{\prime} = GRU^1(y_{j-1}, s_{j-1})
            _, state_prime = self.cell1(inputs, state, scope="gru1")
            # c_j = att(H, s_j^{\prime})
            alpha = attention(state_prime, keys, key_mask, self.dim_hid, self.dim_key)
            context = T.sum(alpha[:, :, None] * values, 0)
            # s_j = GRU^2(c_j, s_j^{\prime})
            output, next_state = self.cell2(context, state_prime, scope="gru2")
            next_state = (1.0 - mask) * state + mask * next_state
            return [alpha, next_state]

        with ops.variable_scope("decoder"):
            seq = [target_inputs, tgt_mask]
            outputs_info = [None, initial_state]
            nonseq = [keys, values, src_mask]
            (alpha, state), updaptes = theano.scan(attention_loop, seq,
                                                   outputs_info, nonseq)
            attention_score = alpha

        alignment_inputs = [src_seq, src_mask, tgt_seq, tgt_mask]
        alignment_outputs = attention_score
        align = theano.function(alignment_inputs, alignment_outputs)
        return align

    def prediction(self, y_emb, state, context, y_pos, keep_prob=1.0):
        """
        readout -> softmax
        p(y_j) \propto f(y_{j-1}, s_{j}, c_{j})
        :param y_pos:
        :param y_emb:
        :param state: 
        :param context: 
        :param keep_prob: 
        :return: 
        """
        state = nn.feedforward([state, y_pos], [[self.dim_hid, self.poshdim], self.dim_hid], True,
                               activation=T.tanh, scope="enhancedstate")
        
        features = [state, y_emb, context, y_pos]
        readout = nn.feedforward(features, [[self.dim_hid, self.dim_y, self.dim_value, self.poshdim], self.dim_readout], True,
                                 activation=T.tanh,
                                 scope="readout")

        if keep_prob < 1.0:
             readout = nn.dropout(readout, keep_prob=keep_prob)
        logits = nn.linear(readout, [self.dim_readout, self.n_y_vocab], True,
                           scope="logits")

        if logits.ndim == 3:
            new_shape = [logits.shape[0] * logits.shape[1], -1]
            logits = logits.reshape(new_shape)

        probs = T.nnet.softmax(logits)
        return probs

    def forward(self, y_seq, y_emb, mask, cwkeys, cpkeys, fwkeys, fpkeys, key_mask, values, initial_state, pos_seq, keep_prob=1.0):
        # shift embedding
        y_shifted = T.zeros_like(y_emb)
        y_shifted = T.set_subtensor(y_shifted[1:], y_emb[:-1])
        y_emb = y_shifted

        # feed
        states, contexts, y_pos_states, posscores = Decoder.scan(self, y_emb, mask, cwkeys, cpkeys, fwkeys, fpkeys,
                                                                 key_mask, values, initial_state)
        # p(y_j) \propto f(y_{j-1}, s_{j}, c_{j})
        # new: p(y_j) \propto f(y_{j-1}, s_{j}, c_{j}, y_pos_{j})
        probs = self.prediction(y_emb, states, contexts, y_pos_states, keep_prob)
        # compute cost
        cost, snt_cost = self.get_cost(y_seq, mask, probs)

        new_shape = [posscores.shape[0] * posscores.shape[1], -1]
        posprobs = posscores.reshape(new_shape)
        # posprobs = T.nnet.softmax(posscores)

        poscost, _ = self.get_cost(pos_seq, mask, posprobs)

        return states, contexts, cost, snt_cost, poscost

