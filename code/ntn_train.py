# std
import random
import datetime

# internal
import ntn_input
import ntn
import ntn_plot
import params

# 3rd party
import tensorflow as tf
import numpy as np
import numpy.matlib


def data_to_indexed_train(data, entities, relations):
  entity_to_index = {entities[i]: i for i in range(len(entities))}
  relation_to_index = {relations[i]: i for i in range(len(relations))}
  indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],
                   entity_to_index[data[i][2]]) for i in range(len(data))]
  return indexed_data


def data_to_indexed_eval(data, entities, relations):
  entity_to_index = {entities[i]: i for i in range(len(entities))}
  relation_to_index = {relations[i]: i for i in range(len(relations))}
  # indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],
  #                  entity_to_index[data[i][2]], float(data[i][3])) for i in range(len(data))]
  indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],
                   entity_to_index[data[i][2]], 1.0) for i in range(len(data))]
  return indexed_data


def data_to_indexed_validation(data, entities, relations):
  entity_to_index = {entities[i]: i for i in range(len(entities))}
  relation_to_index = {relations[i]: i for i in range(len(relations))}
  indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],
                   entity_to_index[data[i][2]], float(data[i][3])) for i in range(len(data))]
  return indexed_data


def get_batch(batch_size, data, num_entities, corrupt_size):
  random_indices = random.sample(range(len(data)), batch_size)
  # random_indices = list(range(len(data)))
  # random.shuffle(random_indices)
  # data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
  batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities - 1))
           for i in random_indices for j in range(corrupt_size)]
  return batch, random_indices


def split_batch(data_batch, num_relations):
  batches = [[] for i in range(num_relations)]
  for e1, r, e2, e3 in data_batch:
    batches[r].append((e1, e2, e3))
  return batches


def split_batch_eval(data, random_indices, num_entities):
  batch = [(data[i][0], data[i][1], data[i][2], data[i][3]) for i in random_indices]
  return batch


def fill_feed_dict_train(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
  feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
  for i in range(len(batch_placeholders)):
    feed_dict[batch_placeholders[i]] = batches[i]
    feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
  return feed_dict


def fill_feed_dict_eval(batches, labels, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
  feed_dict = {corrupt_placeholder: [
      train_both and np.random.random() > 0.5]}
  for i in range(len(batch_placeholders)):
    feed_dict[batch_placeholders[i]] = batches[i]
  for i in range(len(label_placeholders)):
    feed_dict[label_placeholders[i]] = labels[i]
  return feed_dict


# dataset is in the form (e1, R, e2, label)
def data_to_relation_sets(data_batch, num_relations):
  batches = [[] for i in range(num_relations)]
  labels = [[] for i in range(num_relations)]
  for e1, r, e2, label in data_batch:
    batches[r].append((e1, e2, 1))
    labels[r].append([label])
  return (batches, labels)


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
        num_epochs=100):

  print("Begin!")
  # python list of (e1, R, e2) for entire training set in string form
  print("Load training data...")
  raw_training_data = ntn_input.load_training_data(params.data_path)
  print("Load entities and relations...")
  entities_list = ntn_input.load_entities(params.data_path)
  relations_list = ntn_input.load_relations(params.data_path)
  # python list of (e1, R, e2) for entire training set in index form
  indexed_training_data = data_to_indexed_train(raw_training_data, entities_list, relations_list)
  indexed_eval_data = data_to_indexed_eval(raw_training_data, entities_list, relations_list)

  print("Load validation data...")
  validation_data = ntn_input.load_dev_data(params.data_path)
  print("Load entities and relations...")
  entities_list = ntn_input.load_entities(params.data_path)
  relations_list = ntn_input.load_relations(params.data_path)
  # python list of (e1, R, e2) for entire training set in index form
  indexed_validation_data = data_to_indexed_validation(validation_data, entities_list, relations_list)

  batch_size_validation = len(indexed_validation_data)
  # indexed_eval_data = data_to_indexed_eval(raw_training_data, entities_list, relations_list)

  print("Load embeddings...")
  (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)

  num_entities = len(entities_list)
  num_relations = len(relations_list)

  num_iters = num_epochs
  batch_size = batch_size
  # corrupt_size = params.corrupt_size
  # slice_size = params.slice_size
  corrupt_size = corrupt_size
  slice_size = slice_size

  with tf.Graph().as_default():
    print("Starting to build graph " + str(datetime.datetime.now()))
    batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_' + str(i)) for i in range(num_relations)]
    label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_' + str(i)) for i in range(num_relations)]

    corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))  # Which of e1 or e2 to corrupt?
    inference_train, inference_eval = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
                                                    num_entities, num_relations, slice_size, batch_size, False, label_placeholders)
    # loss = ntn.loss(inference_train, params.regularization)
    # training = ntn.training(loss, params.learning_rate)

    loss = ntn.loss(inference_train, l2_lambda)
    training = ntn.training(loss, lr, mom_coeff)

    # inference_eval = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec,
    #                                num_entities, num_relations, slice_size, batch_size, True, label_placeholders)

    eval_correct = ntn.eval(inference_eval)

    validation_correct = ntn.eval(inference_eval)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver(tf.trainable_variables())
    prev_accuracy_validation = 0

    # Save the training loss and accuracies on training and validation data.
    train_costs = list()
    train_accs = list()
    val_costs = list()
    val_accs = list()

    for i in range(1, num_epochs + 1):
      print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
      data_batch, random_indices = get_batch(batch_size, indexed_training_data, num_entities, corrupt_size)
      relation_batches = split_batch(data_batch, num_relations)
      eval_batch = split_batch_eval(indexed_eval_data, random_indices, num_entities)
      eval_batches, eval_labels = data_to_relation_sets(eval_batch, num_relations)

      if i % params.save_per_iter == 0:
        saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

      feed_dict = fill_feed_dict_train(relation_batches, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder)
      _, cost_training = sess.run([training, loss], feed_dict=feed_dict)

      accuracy_training = do_eval(sess, eval_correct, batch_placeholders, label_placeholders, corrupt_placeholder, eval_batches, eval_labels, batch_size)
      print('training cost:', cost_training)
      print('training accuracy:', accuracy_training)
      print()

      train_costs.append((i, cost_training))
      train_accs.append((i, accuracy_training))

      cost_validation = None
      accuracy_validation = None

      # TODO: Eval against dev set?
      if i % val_per_iter == 0 or i == 1:
        print("Beginning building validation")
        # data_batch = get_batch_val(batch_size, indexed_training_data, num_entities, corrupt_size)
        # # relation, e1s, e2s, e_corrupts, targets
        # relation_batches = split_batch(data_batch, num_relations)
        # feed_dict = fill_feed_dict(relation_batches, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder, num_relations)

        batches_validation, labels_validation = data_to_relation_sets(indexed_validation_data, num_relations)

        # cost_val, acc_val = sess.run([loss_validation, accuracy_validation], feed_dict=feed_dict)
        loss_validation = ntn.loss(inference_train, params.regularization)
        feed_dict = fill_feed_dict_train(batches_validation, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder)
        cost_validation = sess.run([loss_validation], feed_dict=feed_dict)
        accuracy_validation = do_eval(sess, validation_correct, batch_placeholders, label_placeholders, corrupt_placeholder, batches_validation, labels_validation, batch_size)
        print('validation cost:', cost_validation)
        print('validation accuracy:', accuracy_validation)
        print()

        val_costs.append((i, cost_validation))
        val_accs.append((i, accuracy_validation))

        # early stopping
        if accuracy_validation <= prev_accuracy_validation and i > 100 and stop_early:
          print("Validation accuracy stopped improving, stopping training early after %d epochs!" % i)
          print()
          break

        prev_accuracy_validation = accuracy_validation

    print("check pointing model...")
    saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')
    print("model checkpoint complete!")

  return train_costs, train_accs, val_costs, val_accs


def do_eval(sess, eval_correct, batch_placeholders, label_placeholders, corrupt_placeholder, eval_batches, eval_labels, num_examples):
  print("Starting do eval")
  true_count = 0.

  feed_dict = fill_feed_dict_eval(eval_batches, eval_labels, params.train_both,
                                  batch_placeholders, label_placeholders, corrupt_placeholder)
  # predictions,labels = sess.run(eval_correct, feed_dict)
  predictions, labels = sess.run(eval_correct, feed_dict)

  for i in range(len(predictions[0])):
    if predictions[0][i] > 0 and labels[0][i] == 1:
      true_count += 1.0
    elif predictions[0][i] < 0 and labels[0][i] == -1:
      true_count += 1.0

  precision = float(true_count) / float(len(predictions[0]))
  return precision


def main(argv):
  run_training()


if __name__ == "__main__":
  tf.app.run()
