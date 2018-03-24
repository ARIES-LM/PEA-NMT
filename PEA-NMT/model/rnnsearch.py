import numpy
import theano
import theano.sandbox.rng_mrg
import theano.tensor as T

import nn
import ops
from bridge import map_key
from encoder import Encoder
from decoder import DecoderGruCond
from search import beam, select_nbest


class rnnsearch:
    def __init__(self, **option):
        # source and target embedding dim
        sedim, tedim, xposhdim, yposhdim = option["embdim"]
        # source, target and attention hidden dim
        shdim, thdim, ahdim, xposnn, yposnn, word2pos, pos2word, pos2pos = option["hidden"]
        # maxout hidden dim
        maxdim = option["maxhid"]
        # maxout part
        maxpart = option["maxpart"]
        # deepout hidden dim
        deephid = option["deephid"]
        svocab, tvocab, tagvocab = option["vocabulary"]
        sw2id, sid2w = svocab
        tw2id, tid2w = tvocab

        stag2id, ttag2id = tagvocab
        # source and target vocabulary size
        svsize, tvsize = len(sid2w), len(tid2w)
        stagsize, ttagsize = len(stag2id), len(ttag2id)

        if "scope" not in option or option["scope"] is None:
            option["scope"] = "rnnsearch"

        if "initializer" not in option:
            option["initializer"] = None

        if "regularizer" not in option:
            option["regularizer"] = None

        if "keep_prob" not in option:
            option["keep_prob"] = 1.0

        dtype = theano.config.floatX
        initializer = option["initializer"]
        regularizer = option["regularizer"]
        keep_prob = option["keep_prob"] or 1.0

        scope = option["scope"]
        decoder_scope = "decoder"

        encoder = Encoder(sedim, shdim)
        decoderType = eval("Decoder{}".format(option["decoder"]))
        decoder = decoderType(tedim, thdim, ahdim, 2 * shdim + xposhdim, dim_maxout=maxdim, max_part=maxpart, dim_readout=deephid,
                              n_y_vocab=tvsize, n_y_tagvocab=ttagsize, poshdim=yposhdim, posnndim=yposnn,
                              word2pos=word2pos, pos2word=pos2word, pos2pos=pos2pos)

        # training graph
        with ops.variable_scope(scope, initializer=initializer,
                                regularizer=regularizer, dtype=dtype):
            src_seq = T.imatrix("source_sequence")
            src_mask = T.matrix("source_sequence_mask")
            tgt_seq = T.imatrix("target_sequence")
            tgt_mask = T.matrix("target_sequence_mask")
            src_pos = T.imatrix("source_postag")
            tgt_pos = T.imatrix("target_postag")

            with ops.variable_scope("source_embedding"):
                source_embedding = ops.get_variable("embedding",
                                                    [svsize, sedim])
                source_bias = ops.get_variable("bias", [sedim])

            with ops.variable_scope("target_embedding"):
                target_embedding = ops.get_variable("embedding",
                                                    [tvsize, tedim])
                target_bias = ops.get_variable("bias", [tedim])

            with ops.variable_scope("srctag_embedding"):
                srctag_embedding = ops.get_variable("embedding", [stagsize, xposhdim])
                srctag_bias = ops.get_variable("bias", [xposhdim])

            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)

            source_inputs = source_inputs + source_bias
            target_inputs = target_inputs + target_bias

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            with ops.variable_scope("srcpostagger"):
                tempstates = nn.feedforward(annotation, [shdim*2, xposnn], True,
                                                 scope="staggerstates", activation=T.nnet.relu)
                scores = nn.linear(tempstates, [xposnn, stagsize], True,
                                            scope="staggerscores")

                new_shape = [scores.shape[0] * scores.shape[1], -1]
                scores = scores.reshape(new_shape)
                srcposprobs = T.nnet.softmax(scores)

                srctaggerstates = T.dot(srcposprobs, srctag_embedding) + srctag_bias
                srctaggerstates = srctaggerstates.reshape([annotation.shape[0], annotation.shape[1], -1])

                idx = T.arange(src_pos.flatten().shape[0])
                ce = -T.log(srcposprobs[idx, src_pos.flatten()])
                ce = ce.reshape(src_pos.shape)
                ce = T.sum(ce * src_mask, 0)
                srcpos_cost = T.mean(ce)

            tempposkeys = T.concatenate([srctaggerstates, tempstates], -1)

            src_words_keys = map_key(annotation, 2 * shdim, ahdim, "srcwordkeys")
            src_pos_keys = map_key(tempposkeys, xposnn + xposhdim, word2pos, "srcposkeys")

            pos_words_keys = map_key(annotation, 2 * shdim, pos2word, "pos2wordkeys")
            pos_pos_keys = map_key(tempposkeys, xposnn + xposhdim, pos2pos, "pos2poskeys")

            annotation = T.concatenate([annotation, srctaggerstates], -1)

            # compute initial state for decoder
            # first state of backward encoder
            final_state = T.concatenate([r_states[0], srctaggerstates[0]], -1)
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim + xposhdim, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)

                _, _, transcost, _, tgtpos_cost = decoder.forward(tgt_seq, target_inputs, tgt_mask, src_words_keys, src_pos_keys,
                                                                  pos_words_keys, pos_pos_keys, src_mask,
                                                                annotation, initial_state, tgt_pos, keep_prob)

        lambx = theano.shared(numpy.asarray(option["lambda"][0], dtype), "lambdax")
        lamby = theano.shared(numpy.asarray(option["lambda"][1], dtype), "lambday")

        totalcost = transcost + lambx * srcpos_cost + lamby * tgtpos_cost
        training_inputs = [src_seq, src_mask, tgt_seq, tgt_mask, src_pos, tgt_pos]
        training_outputs = [srcpos_cost, tgtpos_cost, transcost, totalcost]

        # decoding graph
        with ops.variable_scope(scope, reuse=True):
            prev_words = T.ivector("prev_words")

            source_inputs = nn.embedding_lookup(source_embedding, src_seq)
            source_inputs = source_inputs + source_bias
            target_inputs = nn.embedding_lookup(target_embedding, tgt_seq)
            target_inputs = target_inputs + target_bias

            states, r_states = encoder.forward(source_inputs, src_mask)
            annotation = T.concatenate([states, r_states], 2)

            with ops.variable_scope("srcpostagger"):
                tempstates = nn.feedforward(annotation, [shdim*2, xposnn], True,
                                                 scope="staggerstates", activation=T.nnet.relu)
                scores = nn.linear(tempstates, [xposnn, stagsize], True,
                                            scope="staggerscores")

                new_shape = [scores.shape[0] * scores.shape[1], -1]
                scores = scores.reshape(new_shape)
                srcposprobs = T.nnet.softmax(scores)

                srctaggerstates = T.dot(srcposprobs, srctag_embedding) + srctag_bias
                srctaggerstates = srctaggerstates.reshape([annotation.shape[0], annotation.shape[1], -1])

            tempposkeys = T.concatenate([srctaggerstates, tempstates], -1)

            src_words_keys = map_key(annotation, 2 * shdim, ahdim, "srcwordkeys")
            src_pos_keys = map_key(tempposkeys, xposnn + xposhdim, word2pos, "srcposkeys")

            pos_words_keys = map_key(annotation, 2 * shdim, pos2word, "pos2wordkeys")
            pos_pos_keys = map_key(tempposkeys, xposnn + xposhdim, pos2pos, "pos2poskeys")

            annotation = T.concatenate([annotation, srctaggerstates], -1)

            # decoder
            final_state = T.concatenate([r_states[0], srctaggerstates[0]], -1)
            with ops.variable_scope(decoder_scope):
                initial_state = nn.feedforward(final_state, [shdim+xposhdim, thdim],
                                               True, scope="initial",
                                               activation=T.tanh)

            prev_inputs = nn.embedding_lookup(target_embedding, prev_words)
            prev_inputs = prev_inputs + target_bias

            cond = T.neq(prev_words, 0)
            # zeros out embedding if y is 0, which indicates <s>
            prev_inputs = prev_inputs * cond[:, None]

            with ops.variable_scope(decoder_scope):
                mask = T.ones_like(prev_words, dtype=dtype)
                next_state, context, next_pos, tgtposprob = decoder.step(prev_inputs, mask, initial_state,
                                                                         src_words_keys, src_pos_keys,
                                                                         pos_words_keys, pos_pos_keys,
                                                                        annotation, src_mask)
                if option["decoder"] == "GruSimple":
                    probs = decoder.prediction(prev_inputs, initial_state, context)
                elif option["decoder"] == "GruCond":
                    probs = decoder.prediction(prev_inputs, next_state, context, next_pos)

        # encoding
        encoding_inputs = [src_seq, src_mask]
        encoding_outputs = [annotation, initial_state, src_words_keys, src_pos_keys, pos_words_keys, pos_pos_keys, srcposprobs]
        encode = theano.function(encoding_inputs, encoding_outputs)

        if option["decoder"] == "GruSimple":
            prediction_inputs = [prev_words, initial_state, annotation,
                                 mapped_keys, src_mask]
            prediction_outputs = [probs, context]
            predict = theano.function(prediction_inputs, prediction_outputs)

            generation_inputs = [prev_words, initial_state, context]
            generation_outputs = next_state
            generate = theano.function(generation_inputs, generation_outputs)

            self.predict = predict
            self.generate = generate
        elif option["decoder"] == "GruCond":
            prediction_inputs = [prev_words, initial_state, annotation,
                                 src_words_keys, src_pos_keys, pos_words_keys, pos_pos_keys, src_mask]
            prediction_outputs = [probs, next_state, tgtposprob]
            predict = theano.function(prediction_inputs, prediction_outputs, on_unused_input='warn')
            self.predict = predict

        self.cost = totalcost
        self.inputs = training_inputs
        self.outputs = training_outputs
        self.updates = []
        self.encode = encode
        self.option = option


def beamsearch(models, seq, mask=None, beamsize=10, normalize=False,
               maxlen=None, minlen=None, arithmetic=False, dtype=None, suppress_unk=False):
    dtype = dtype or theano.config.floatX

    if not isinstance(models, (list, tuple)):
        models = [models]

    num_models = len(models)

    # get vocabulary from the first model
    option = models[0].option
    vocab = option["vocabulary"][1][1]
    eosid = option["eosid"]
    bosid = option["bosid"]
    unk_sym = models[0].option["unk"]
    unk_id = option["vocabulary"][1][0][unk_sym]

    if maxlen is None:
        maxlen = seq.shape[0] * 3

    if minlen is None:
        minlen = seq.shape[0] / 2

    # encoding source
    if mask is None:
        mask = numpy.ones(seq.shape, dtype)

    outputs = [model.encode(seq, mask) for model in models]

    annotations, states, coarse_words_keys, coarse_pos_keys, mapped_fineword_annots, mapped_finepos_annots = [], [], [], [], [], []
    for item in outputs:
        annotations.append(item[0])
        states.append(item[1])
        coarse_words_keys.append(item[2])
        coarse_pos_keys.append(item[3])
        mapped_fineword_annots.append(item[4])
        mapped_finepos_annots.append(item[5])

    initial_beam = beam(beamsize)
    size = beamsize
    # bosid must be 0
    initial_beam.candidates = [[bosid]]
    initial_beam.scores = numpy.zeros([1], dtype)

    hypo_list = []
    beam_list = [initial_beam]
    done_predicate = lambda x: x[-1] == eosid

    for k in range(maxlen):
        # get previous results
        prev_beam = beam_list[-1]
        candidates = prev_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda cand: cand[-1], candidates), "int32")

        # compute context first, then compute word distribution
        batch_mask = numpy.repeat(mask, num, 1)
        batch_annots = map(numpy.repeat, annotations, [num] * num_models,
                           [1] * num_models)

        batch_mcoarseword = map(numpy.repeat, coarse_words_keys, [num] * num_models,
                            [1] * num_models)
        batch_mcoarsepos = map(numpy.repeat, coarse_pos_keys, [num] * num_models,
                            [1] * num_models)

        batch_mfineword = map(numpy.repeat, mapped_fineword_annots, [num] * num_models,
                            [1] * num_models)
        batch_mfinepos = map(numpy.repeat, mapped_finepos_annots, [num] * num_models,
                            [1] * num_models)

        # predict returns [probs, next_state, next_pos]
        outputs = [model.predict(last_words, state, annot, coarsemword, coarsempos, finemword, finempos, batch_mask)
                   for model, state, annot, coarsemword, coarsempos, finemword, finempos in
                   zip(models, states, batch_annots,
                       batch_mcoarseword, batch_mcoarsepos, batch_mfineword, batch_mfinepos)]
        prob_dists = [item[0] for item in outputs]

        # search nbest given word distribution
        if arithmetic:
            logprobs = numpy.log(sum(prob_dists) / num_models)
        else:
            # geometric mean
            logprobs = sum(numpy.log(prob_dists)) / num_models

        if suppress_unk:
            logprobs[:, unk_id] = -numpy.inf

        if k < minlen:
            logprobs[:, eosid] = -numpy.inf  # make sure eos won't be selected

        # force to add eos symbol
        if k == maxlen - 1:
            # copy
            eosprob = logprobs[:, eosid].copy()
            logprobs[:, :] = -numpy.inf
            logprobs[:, eosid] = eosprob  # make sure eos will be selected

        next_beam = beam(size)
        finished, remain_beam_indices = next_beam.prune(logprobs, done_predicate, prev_beam)

        hypo_list.extend(finished)  # completed translation
        size -= len(finished)

        if size == 0:  # reach k completed translation before maxlen
            break

        # generate next state
        candidates = next_beam.candidates
        num = len(candidates)
        last_words = numpy.array(map(lambda t: t[-1], candidates), "int32")

        if option["decoder"] == "GruSimple":
            contexts = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model
            contexts = select_nbest(contexts, remain_beam_indices)

            states = [model.generate(last_words, state, context)
                      for model, state, context in zip(models, states, contexts)]
        elif option["decoder"] == "GruCond":
            states = [item[1] for item in outputs]
            states = select_nbest(states, remain_beam_indices)  # select corresponding states for each model
        beam_list.append(next_beam)

    # postprocessing
    if len(hypo_list) == 0:
        score_list = [0.0]
        hypo_list = [[eosid]]
    else:
        score_list = [item[1] for item in hypo_list]
        # exclude bos symbol
        hypo_list = [item[0][1:] for item in hypo_list]

    for i, (trans, score) in enumerate(zip(hypo_list, score_list)):
        count = len(trans)
        if count > 0:
            if normalize:
                score_list[i] = score / count
            else:
                score_list[i] = score

    # sort
    hypo_list = numpy.array(hypo_list)[numpy.argsort(score_list)]
    score_list = numpy.array(sorted(score_list))

    output = []

    for trans, score in zip(hypo_list, score_list):
        trans = map(lambda x: vocab[x], trans)
        output.append((trans, score))

    return output
