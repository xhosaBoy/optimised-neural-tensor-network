# std
import sys
import random
import logging

# 3rd party
import tensorflow as tf
import numpy as np
import numpy.matlib

# internal
import ntn_input
import ntn
import params

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i] : i for i in range(len(entities))}
    relation_to_index = {relations[i] : i for i in range(len(relations))}

    # build subject, predict, object
    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]]) for i in range(len(data))]

    return indexed_data


def data_to_indexed_eval(data, entities, relations):

    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]], 1.0) for i in range(len(data))]

    return indexed_data


def get_batch(batch_size, data, num_entities, corrupt_size, idx=None):

    idx = idx if idx else 0
    # random_indices = random.sample(range(len(data)), batch_size)
    indices = list(range(idx, min(idx + batch_size, len(data))))
    batch = [(data[i][0],
              data[i][1],
              data[i][2],
              random.randint(0, num_entities - 1)) for i in indices for j in range(corrupt_size)]

    logger.debug(f'batch: {batch}')

    return batch, indices


def split_batch(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]

    for e1, r, e2, e3 in data_batch:
        batches[r].append((e1, e2, e3))

    return batches


def split_batch_eval(data, indices, num_entities):

    batch = [(data[i][0], data[i][1], data[i][2], data[i][3]) for i in indices]

    return batch


def data_to_relation_sets(data_batch, num_relations):

    batches = [[] for i in range(num_relations)]
    labels = [[] for i in range(num_relations)]

    for e1, r, e2, label in data_batch:
        batches[r].append((e1, e2, 1))
        labels[r].append([label])

    return batches, labels


def fill_feed_dict(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}

    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i]
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]

    return feed_dict


def fill_feed_dict_eval(batches, labels, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):

    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}

    logger.debug(f'batches: {batches}')

    for i in range(len(batch_placeholders)):
        if batches[i]:
            feed_dict[batch_placeholders[i]] = batches[i]
            logger.debug(f'Placed the batch: {batches[i]}')
        else:
            batch = [(0, 0, 0)]
            feed_dict[batch_placeholders[i]] = batch
            logger.debug(f'PLACED DEFAULT BATCH: {batch}')

    for i in range(len(label_placeholders)):
        if batches[i]:
            feed_dict[label_placeholders[i]] = labels[i]
            logger.debug(f'Placed the labels: {labels[i]}')
        else:
            label = [[0]]
            feed_dict[label_placeholders[i]] = label
            logger.debug(f'Placed the labels: {labels}')

    return feed_dict


def do_eval(sess,
            eval_correct,
            batch_placeholders,
            label_placeholders,
            corrupt_placeholder,
            eval_batches,
            eval_labels,
            num_examples):

    logger.info("Starting do eval...")
    true_count = 0.

    feed_dict = fill_feed_dict_eval(eval_batches,
                                    eval_labels,
                                    params.train_both,
                                    batch_placeholders,
                                    label_placeholders,
                                    corrupt_placeholder)

    predictions, labels = sess.run(eval_correct, feed_dict)

    for i in range(len(predictions[0])):
        if predictions[0][i] > 0 and labels[0][i] == 1:
            true_count += 1.0
        elif predictions[0][i] < 0 and labels[0][i] == -1:
            true_count += 1.0

    precision = float(true_count) / float(len(predictions[0]))

    return precision


def run_training():
    logger.info("Begin!")
    # python list of (e1, R, e2) for entire training set in string form
    logger.info("Load training data...")
    raw_training_data = ntn_input.load_training_data(params.data_path)

    logger.info("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    # python list of (e1, R, e2) for entire training set in index form
    # subject, predicate, object
    indexed_training_data = data_to_indexed(raw_training_data, entities_list, relations_list)
    indexed_eval_data = data_to_indexed_eval(raw_training_data, entities_list, relations_list)

    logger.info("Load embeddings...")
    # wordvecs, ids
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        logger.info(f'Starting to build graph...')
        batch_placeholders = [tf.placeholder(tf.int32,
                                             shape=(None, 3),
                                             name='batch_' + str(i)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32,
                                             shape=(None, 1),
                                             name='label_' + str(i)) for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) # Which of e1 or e2 to corrupt?

        inference = ntn.inference(batch_placeholders,
                                  corrupt_placeholder,
                                  init_word_embeds,
                                  entity_to_wordvec,
                                  num_entities,
                                  num_relations,
                                  slice_size,
                                  batch_size,
                                  False,
                                  label_placeholders)
        loss = ntn.loss(inference, params.regularization)
        training = ntn.training(loss, params.learning_rate)
        # evaluate = ntn.eval(inference)
        eval_correct = ntn.eval(inference)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())

        for i in range(1, num_iters):
            logger.info(f'Starting iter {i}')
            # randomised subjects, predicates, objects, for given predicate
            # data_batch = get_batch(batch_size, indexed_training_data, num_entities, corrupt_size)
            data_batch, indices = get_batch(batch_size,
                                            indexed_training_data,
                                            num_entities,
                                            corrupt_size,
                                            i)

            # relation, e1s, e2s, e_corrupts
            relation_batches = split_batch(data_batch, num_relations)

            eval_batch = split_batch_eval(indexed_eval_data, indices, num_entities)
            eval_batches, eval_labels = data_to_relation_sets(eval_batch, num_relations)

            if i % params.save_per_iter == 0:
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            feed_dict = fill_feed_dict(relation_batches,
                                       params.train_both,
                                       batch_placeholders,
                                       label_placeholders,
                                       corrupt_placeholder)

            _, cost_training = sess.run([training, loss], feed_dict=feed_dict)
            accuracy_training = do_eval(sess,
                                        eval_correct,
                                        batch_placeholders,
                                        label_placeholders,
                                        corrupt_placeholder,
                                        eval_batches,
                                        eval_labels,
                                        batch_size)
            # _, cost_training, (score_pos, score_neg) = sess.run([training, loss, evaluate], feed_dict=feed_dict)
            # print("Loss: ", cost_training, "score_pos, score_neg: ", score_pos, score_neg)

            logger.info(f'epoch: {i + 1}, cost_training: {cost_training}')
            logger.info(f'epoch: {i + 1}, accuracy_training: {accuracy_training}')

            #TODO: Eval against dev set?


def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()
