import tensorflow as tf
import params
import ntn_input
import random
from collections import defaultdict

# Inference
# Loss
# Training

# returns a (batch_size*corrupt_size, 2) vector corresponding to [g(T^i), g(T_c^i)] for all i
def inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec,\
        num_entities, num_relations, slice_size, batch_size, is_eval, label_placeholders):
    print("Beginning building inference:")
    #TODO: We need to check the shapes and axes used here!
    print("Creating variables")
    d = 100 # embed_size
    k = slice_size
    ten_k = tf.constant([k])
    num_words = len(init_word_embeds)
    E = tf.Variable(init_word_embeds) # d=embed size
    W = [tf.Variable(tf.truncated_normal([d, d, k])) for r in range(num_relations)]
    print('W:', W[0])
    V = [tf.Variable(tf.zeros([k, 2 * d])) for r in range(num_relations)]
    b = [tf.Variable(tf.zeros([k, 1])) for r in range(num_relations)]
    U = [tf.Variable(tf.ones([1, k])) for r in range(num_relations)]

    print("Create entity word vec IDs")
    # python list of tf vectors: i -> list of word indices cooresponding to entity i
    ent2word = [tf.constant(entity_i) - 1 for entity_i in entity_to_wordvec]

    #(num_entities, d) matrix where row i cooresponds to the entity embedding (word embedding average) of entity i
    print("Calcing entEmbed...")
    # entword id, gather wordvec paramaters to update
    # select only word embeddings we are interested in
    entEmbed = tf.pack([tf.reduce_mean(tf.gather(E, entword), 0) for entword in ent2word])
    # subset of all words
    print(entEmbed.get_shape())


###################################################### Extract Entity Relationship Triples ###################################################


    # e1s, e2s, e_corrupts, relations
    e1, e2, e3, relation = tf.split(1, 4, tf.cast(batch_placeholders, tf.int32)) #TODO: should the split dimension be 0 or 1?
    # combine wordvec id, wordvec parameters, and wordvec
    e1v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e1, name='e1v'), [1]))
    print('e1v:', e1v)
    e2v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e2, name='e2v'), [1]))
    print('e2v:', e2v)
    # e3v = tf.transpose(tf.squeeze(tf.gather(entEmbed, e3, name='e3v' + str(r)), [1]))
    e1v_pos = e1v
    e2v_pos = e2v
    # e1v_neg = e1v
    # e2v_neg = e3v
    num_rel_r = tf.expand_dims(tf.shape(e1v_pos)[1], 0)
    print('num_rel_r:', num_rel_r)


##########################################################################################################################################


    # predictions = list()
    logits = list()
    targets = list()

    # recursive neural network
    print("Beginning relations loop")
    for r in range(num_relations):
        print("Relations loop " + str(r))
        # print("Starting preactivation funcs")

        preactivation_pos = list()
        # preactivation_neg = list()
        entity_dict = defaultdict(list)

        for slice in range(k):
            print('W[r][:, :, slice]:', W[r][:, :, slice])
            print('e2v_pos:', e2v_pos)
            print('e1v_pos:', e1v_pos)
            preactivation_pos.append(tf.reduce_sum(e1v_pos * tf.matmul(W[r][:, :, slice], e2v_pos), 0))
            print('preactivation_pos:', preactivation_pos)
            # preactivation_neg.append(tf.reduce_sum(e1v_neg * tf.matmul(W[r][:, :, slice], e2v_neg), 0))

        preactivation_pos = tf.pack(preactivation_pos)
        print('preactivation_pos pack:', preactivation_pos)
        # preactivation_neg = tf.pack(preactivation_neg)

        temp2_pos = tf.matmul(V[r], tf.concat(0, [e1v_pos, e2v_pos]))
        # temp2_neg = tf.matmul(V[r], tf.concat(0, [e1v_neg, e2v_neg]))

        #print("   temp2_pos: "+str(temp2_pos.get_shape()))
        preactivation_pos = preactivation_pos + temp2_pos + b[r]
        print('preactivation_pos:', preactivation_pos)
        # preactivation_neg = preactivation_neg + temp2_neg + b[r]

        #print("Starting activation funcs")
        #activation_pos = tf.tanh(preactivation_pos)
        activation_pos = tf.sigmoid(preactivation_pos)
        # activation_pos = tf.nn.relu(preactivation_pos)
        # activation_pos = tf.minimum(activation_pos, 1)
        # activation_neg = tf.tanh(preactivation_neg)
        # print("activation_pos: " + str(activation_pos.get_shape()))

        score_pos = tf.reshape(tf.matmul(U[r], activation_pos), num_rel_r)
        # score_pos = tf.reshape(tf.sigmoid(tf.matmul(U[r], preactivation_pos)), num_rel_r)
        # score_neg = tf.reshape(tf.matmul(U[r], activation_neg), num_rel_r)
        print("score_pos: " + str(score_pos.get_shape()))
        if not is_eval:
            # predictions.append(tf.pack([score_pos, score_neg]))
            # predictions.append(score_pos)
            # print('predictions:', predictions)
            logits.append([score_pos])
            print('logits:', logits)
            targets.append([tf.squeeze(tf.cast(tf.equal(relation, r), tf.float32), [1])])
            print('targets:', targets)
        else:
            # predictions.append(tf.pack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))
            # logit_and_target.append(tf.pack([score_pos, tf.reshape(label_placeholders[r], num_rel_r)]))
            #print("score_pos_and_neg: "+str(predictions[r].get_shape()))
            logits.append([score_pos])
            targets.append([tf.squeeze(tf.cast(tf.equal(relation, r), tf.float32), [1])])

    #print("Concating predictions")
    # predictions = tf.concat(1, predictions)
    # print(predictions[0])
    # logit_and_target = tf.pack(logit_and_target)
    # print('logit_and_target pack:', logit_and_target)
    logits = tf.reshape(tf.pack(logits), [-1, 11])
    print('logits pack:', logits)
    targets = tf.reshape(tf.pack(targets), [-1, 11])
    print('targets pack:', targets)

    return logits, targets


def cross_entropy_tf(predictions):
    """Calculate the cross entropy loss given some model predictions and target (true) values."""
    # targets = tf.one_hot(targets, num_classes)
    # IMPLEMENT-ME: (14)
    # HINT: Have a look at the TensorFlow functions tf.log, tf.reduce_sum and tf.reduce_mean
    cross_entropy = tf.reduce_mean(tf.reduce_sum(-tf.log(predictions)))
    return cross_entropy


def loss(logits, targets, regularization, num_relations, batch_size):

    print("Beginning building loss")
    # temp1 = tf.maximum(tf.sub(predictions[1, :], predictions[0, :]) + 1, 0)
    # temp1 = tf.maximum(predictions[:], 0)
    # temp1 = tf.reduce_sum(temp1)

    # temp2 = tf.sqrt(sum([tf.reduce_sum(tf.square(var)) for var in tf.trainable_variables()]))

    # temp = temp1 + (regularization * temp2)

    # return temp

    # manually construct into entity pair, relationshiship, true-false
    # entity_dict = defaultdict(list)
    # target_dict = defaultdict(list)

    # for r in range(num_relations):
    #   for e in range(batch_size):
    #     entity_dict[e].append(logits_and_targets[r][e][0])
    #     target_dict[e].append(logits_and_targets[r][e][1])
    # # print('entity_dict:', entity_dict)
    # # print('target_dict:', target_dict)

    # # logits, targets = tf.split(1, 2, logits_and_targets)
    # logits = tf.pack(list(entity_dict.values()))
    # print('logits pack:', logits)
    # targets = tf.pack(list(target_dict.values()))
    # print('logits pack:', logits)
    # # predictions = tf.to_float(predictions[0])
    # # targets = tf.reshape(tf.one_hot(tf.cast(targets, tf.int32), num_relations), [1, -1])
    # # targets = tf.cast(targets, tf.int32)

    data_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets))

    # data_loss = cross_entropy_tf(predictions, targets)

    reg_loss = 0.
    for w in tf.trainable_variables():
        reg_loss += tf.nn.l2_loss(w)

    return data_loss + regularization * reg_loss


def training(loss, learningRate):
    print("Beginning building training")

    # return tf.train.AdagradOptimizer(learningRate).minimize(loss)
    return tf.train.AdamOptimizer(learningRate).minimize(loss)
    # return tf.train.AdamOptimizer().minimize(loss)


def eval(logits, my_labels):
    print("Beginning eval")
    # score_pos = tf.reduce_sum(predictions[0, :])
    # score_neg = tf.reduce_sum(predictions[1, :])

    # print("predictions " + str(predictions.get_shape()))
    # inference, labels = tf.split(0, 2, predictions)

    # inference, labels = tf.split(0, 2, predictions)
    #inference = tf.transpose(inference)
    #inference = tf.concat((1-inference), inference)
    #labels = ((tf.cast(tf.squeeze(tf.transpose(labels)), tf.int32))+1)/2
    #print("inference "+str(inference.get_shape()))
    #print("labels "+str(labels.get_shape()))
    # get number of correct labels for the logits (if prediction is top 1 closest to actual)
    #correct = tf.nn.in_top_k(inference, labels, 1)
    predictions = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(my_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # precision = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print('correct_prediction:', correct_prediction)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print('accuracy:', accuracy)
    # precision = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # cast tensor to int and return number of correct labels
    #return tf.reduce_sum(tf.cast(correct, tf.int32))
    # return score_pos, score_neg
    # accuracy = tf.reshape(accuracy, [-1, 1])
    # print('accuracy:', accuracy)

    return accuracy







