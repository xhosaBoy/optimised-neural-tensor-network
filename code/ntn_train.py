# std
import os
import sys
import random
import logging

# internal
import ntn_input
import ntn
import params

# 3rd party
import tensorflow as tf
import numpy as np
import numpy.matlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def data_to_indexed_train(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}

    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]]) for i in range(len(data))]

    return indexed_data


def data_to_indexed_eval(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}

    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]],
                     1.0) for i in range(len(data))]

    return indexed_data


def data_to_indexed_validation(data, entities, relations):
    entity_to_index = {entities[i]: i for i in range(len(entities))}
    relation_to_index = {relations[i]: i for i in range(len(relations))}

    indexed_data = [(entity_to_index[data[i][0]],
                     relation_to_index[data[i][1]],
                     entity_to_index[data[i][2]],
                     float(data[i][3])) for i in range(len(data))]

    return indexed_data


def get_batch(batch_size, data, num_entities, corrupt_size, idx):
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


def fill_feed_dict_train(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    logger.debug(f'batches: {batches}')

    for i in range(len(batch_placeholders)):
        if batches[i]:
            feed_dict[batch_placeholders[i]] = batches[i]
            logger.debug(f'Placed the batch: {batches[i]}')
            feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
        else:
            batch = [(0, 0, 0)]
            feed_dict[batch_placeholders[i]] = batch
            logger.debug(f'PLACED DEFAULT BATCH: {batch}')
            feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[0]))] if len(batches[0]) else [[0.0]]

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


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    labels = [[] for i in range(num_relations)]

    for e1, r, e2, label in data_batch:
        batches[r].append((e1, e2, 1))
        labels[r].append([label])

    return batches, labels


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


def run_training(
        slice_size=10,
        batch_size=10000,
        corrupt_size=10,
        lr=1e-3,
        l2_lambda=1e-4,
        mom_coeff=0.5,
        optimizer_fn=tf.train.MomentumOptimizer(
        learning_rate=1e-3,
        momentum=0.5),
        val_per_iter=10,
        stop_early=True,
        num_epochs=10):

    logger.info("Begin!")
    logger.info("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)

    logger.info("Load training data...")
    raw_training_data = ntn_input.load_training_data(params.data_path)
    np.random.shuffle(raw_training_data)

    # python list of (e1, R, e2) for entire training set in index form
    indexed_training_data = data_to_indexed_train(raw_training_data, entities_list, relations_list)
    indexed_eval_data = data_to_indexed_eval(raw_training_data, entities_list, relations_list)

    logger.info("Load validation data...")
    validation_data = ntn_input.load_dev_data(params.data_path)
    logger.info("Load entities and relations...")
    # python list of (e1, R, e2) for entire training set in index form
    indexed_validation_data = data_to_indexed_validation(validation_data, entities_list, relations_list)

    logger.info("Load test data...")
    test_data = ntn_input.load_test_data(params.data_path)
    logger.info("Load entities and relations...")
    indexed_test_data = data_to_indexed_validation(test_data, entities_list, relations_list)

    logger.info("Load embeddings...")
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    batch_size = batch_size
    corrupt_size = corrupt_size
    slice_size = slice_size

    with tf.Graph().as_default():
        logger.info("Starting to build graph...")
        batch_placeholders = [tf.placeholder(tf.int32,
                                             shape=(None, 3),
                                             name='batch_' + str(i)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32,
                                             shape=(None, 1),
                                             name='label_' + str(i)) for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?

        inference_train, inference_eval = ntn.inference(batch_placeholders,
                                                        corrupt_placeholder,
                                                        init_word_embeds,
                                                        entity_to_wordvec,
                                                        num_entities,
                                                        num_relations,
                                                        slice_size,
                                                        batch_size,
                                                        False,
                                                        label_placeholders)

        loss = ntn.loss(inference_train, l2_lambda)
        training = ntn.training(loss, lr, mom_coeff)

        eval_correct = ntn.eval(inference_eval)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())

        # Save the training loss and accuracies on training and validation data.
        train_costs = list()
        train_accs = list()
        val_costs = list()
        val_accs = list()
        test_costs = list()
        test_accs = list()

        prev_accuracy_validation = 0

        for i in range(1, num_epochs + 1):
            logger.info("Starting EPOCH " + str(i))

            # for j in range(0, len(indexed_training_data), batch_size):
            data_batch, indices = get_batch(batch_size,
                                            indexed_training_data,
                                            num_entities,
                                            corrupt_size,
                                            i)

            relation_batches = split_batch(data_batch, num_relations)
            logger.debug(f'relation_batches: {relation_batches}')

            eval_batch = split_batch_eval(indexed_eval_data, indices, num_entities)
            eval_batches, eval_labels = data_to_relation_sets(eval_batch, num_relations)

            if i % params.save_per_iter == 0:
                saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            feed_dict = fill_feed_dict_train(relation_batches,
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

            train_costs.append((i, cost_training))
            train_accs.append((i, accuracy_training))

            logger.info(f'epoch: {i}, cost_training: {cost_training}')
            logger.info(f'epoch: {i}, accuracy_training: {accuracy_training}')

            accuracy_validation = evaluate(indexed_validation_data,
                                           num_relations,
                                           inference_train,
                                           batch_placeholders,
                                           label_placeholders,
                                           corrupt_placeholder,
                                           sess,
                                           eval_correct,
                                           batch_size,
                                           val_costs,
                                           val_accs,
                                           i)

            accuracy_test = test(indexed_test_data,
                                 num_relations,
                                 inference_train,
                                 batch_placeholders,
                                 label_placeholders,
                                 corrupt_placeholder,
                                 sess,
                                 eval_correct,
                                 batch_size,
                                 test_costs,
                                 test_accs,
                                 i)

        logger.info("check pointing model...")
        checkpoint = os.path.join(params.output_path, f'{params.data_name}_{str(i)}.sess')
        saver.save(sess, checkpoint)
        logger.info("model checkpoint complete!")

    return train_costs, train_accs, val_costs, val_accs


def evaluate(indexed_validation_data,
             num_relations,
             inference_train,
             batch_placeholders,
             label_placeholders,
             corrupt_placeholder,
             sess,
             eval_correct,
             batch_size,
             val_costs,
             val_accs,
             epoch):

    logger.info("Starting VALIDATION " + str(epoch))

    cost_validation = None
    accuracy_validation = None

    batches_validation, labels_validation = data_to_relation_sets(
        indexed_validation_data, num_relations)

    loss_validation = ntn.loss(inference_train, params.regularization)
    feed_dict = fill_feed_dict_train(batches_validation,
                                     params.train_both,
                                     batch_placeholders,
                                     label_placeholders,
                                     corrupt_placeholder)

    cost_validation = sess.run([loss_validation], feed_dict=feed_dict)
    accuracy_validation = do_eval(sess,
                                  eval_correct,
                                  batch_placeholders,
                                  label_placeholders,
                                  corrupt_placeholder,
                                  batches_validation,
                                  labels_validation,
                                  batch_size)

    logger.info(f'validation cost: {cost_validation[0]}')
    logger.info(f'validation accuracy: {accuracy_validation}')

    val_costs.append((epoch, cost_validation))
    val_accs.append((epoch, accuracy_validation))

    return cost_validation[0]


def test(indexed_test_data,
         num_relations,
         inference_train,
         batch_placeholders,
         label_placeholders,
         corrupt_placeholder,
         sess,
         eval_correct,
         batch_size,
         test_costs,
         test_accs,
         epoch):

    logger.info("Starting TESTING")

    cost_test = None
    accuracy_test = None

    batches_test, labels_test = data_to_relation_sets(indexed_test_data, num_relations)

    loss_test = ntn.loss(inference_train, params.regularization)
    feed_dict = fill_feed_dict_train(batches_test,
                                     params.train_both,
                                     batch_placeholders,
                                     label_placeholders,
                                     corrupt_placeholder)

    cost_test = sess.run([loss_test], feed_dict=feed_dict)
    accuracy_test = do_eval(sess,
                            eval_correct,
                            batch_placeholders,
                            label_placeholders,
                            corrupt_placeholder,
                            batches_test,
                            labels_test,
                            batch_size)

    logger.info(f'test cost: {cost_test[0]}')
    logger.info(f'test accuracy: {accuracy_test}')

    test_costs.append((epoch, cost_test))
    test_accs.append((epoch, accuracy_test))

    return cost_test[0]


def main(argv):
    run_training()


if __name__ == "__main__":
    tf.app.run()
