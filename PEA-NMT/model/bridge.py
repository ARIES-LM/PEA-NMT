import theano.tensor as T

import nn
import ops


def map_key(states, dim_state, dim_key, scope=None):
    key = nn.linear(states, [dim_state, dim_key], False, scope=scope or "map-key")
    return key


def attention(query, keys, key_mask, dim_query, dim_key, dtype=None, scope=None):
    with ops.variable_scope(scope or "attention", dtype=dtype):
        # content-based addressing
        # e_i = v_a^T tanh(W query + key_i)
        # alpha = softmax({e_i})
        # (n_query, dim_query) -> (n_query, dim_key)
        mapped_query = nn.linear(query, [dim_query, dim_key], False, scope="map-query")
        # (n_key, n_query, dim_key)
        act = T.tanh(mapped_query[None, :, :] + keys)
        # (n_key, n_query, 1)
        e = nn.linear(act, [dim_key, 1], False, scope="pre-alpha")  # (n_key, n_query, 1)
        # (n_key, n_query)
        e = T.reshape(e, e.shape[:2])
        e = e.T  # (n_query, n_key)
        # match dimension
        key_mask = key_mask.T
        alpha = nn.masked_softmax(e, key_mask)  # (n_query, n_key)
        alpha = alpha.T  # (n_key, n_query)
    return alpha

def fineattention(query, keys, key_mask, dim_query, dim_key, dtype=None, scope=None):
    with ops.variable_scope(scope or "fineattention", dtype=dtype):
        # content-based addressing
        # e_i = v_a^T tanh(W query + key_i)
        # alpha = softmax({e_i})
        # (n_query, dim_query) -> (n_query, dim_key)
        e = []
        for i in range(len(query)):
            obtainkeys = keys[i]
            obtaindimkeys = dim_key[i]
            for j in range(len(obtainkeys)):
                mapped_query = nn.linear(query[i], [dim_query[i], obtaindimkeys[j]], False, scope="map-query_{}{}".format(i, j))
                #(n_key, n_query, dim_key)
                act = T.tanh(mapped_query[None, :, :] + obtainkeys[j])
                # (n_key, n_query, 1)
                em = nn.linear(act, [obtaindimkeys[j], 1], False, scope="pre-alpha_{}{}".format(i, j))  # (n_key, n_query, 1)
                e.append(em)
        e = reduce(T.add, e)
        # (n_key, n_query)
        e = T.reshape(e, e.shape[:2])
        e = e.T  # (n_query, n_key)
        # match dimension
        key_mask = key_mask.T
        alpha = nn.masked_softmax(e, key_mask)  # (n_query, n_key)
        alpha = alpha.T  # (n_key, n_query)
    return alpha

def coarseattention(query, keys, key_mask, dim_query, dim_key, dtype=None, scope=None):
    with ops.variable_scope(scope or "coarseattention", dtype=dtype):
        # content-based addressing
        # e_i = v_a^T tanh(W query + key_i)
        # alpha = softmax({e_i})
        # (n_query, dim_query) -> (n_query, dim_key)

        e = []
        for i in range(len(keys)):
            mapped_query = nn.linear(query, [dim_query, dim_key[i]], False, scope="map-query_{}".format(i))
            # (n_key, n_query, dim_key)
            act = T.tanh(mapped_query[None, :, :] + keys[i])
            # (n_key, n_query, 1)
            em = nn.linear(act, [dim_key[i], 1], False, scope="pre-alpha_{}".format(i))  # (n_key, n_query, 1)
            e.append(em)

        e = reduce(T.add, e)
        # (n_key, n_query)
        e = T.reshape(e, e.shape[:2])
        e = e.T  # (n_query, n_key)
        # match dimension
        key_mask = key_mask.T
        alpha = nn.masked_softmax(e, key_mask)  # (n_query, n_key)
        alpha = alpha.T  # (n_key, n_query)
    return alpha