import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime

def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i] : i for i in range(len(entities))}
    relation_to_index = {relations[i] : i for i in range(len(relations))}
    # build subject, predict, object
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]], \
            entity_to_index[data[i][2]]) for i in range(len(data))]
    # sbujet, predicet , object
    return indexed_data

def get_batch(batch_size, data, num_entities, corrupt_size):
    # return entity 1, entity 2 plus random corrupt entity
    random_indices = random.sample(range(len(data)), batch_size)
    # batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1)) \
    # for i in random_indices for j in range(corrupt_size)]
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1)) \
    for i in range(batch_size) for j in range(corrupt_size)]
    # print('batch sample:', batch[0])
    return batch

def get_batch_val(batch_size, data, num_entities, corrupt_size):
    # return entity 1, entity 2 plus random corrupt entity
    random_indices = random.sample(range(len(data)), batch_size)
    # batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1)) \
    # for i in random_indices for j in range(corrupt_size)]
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1)) \
    for i in range(batch_size + 1, batch_size + 3001) for j in range(corrupt_size)]
    # print('batch sample:', batch[0])
    return batch

def split_batch(data_batch, num_relations):
    # batches = [[] for i in range(num_relations)]
    batches = []
    for e1, r, e2, e3 in data_batch:
        batches.append((e1, e2, e3, r))
    return batches

def fill_feed_dict(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder, num_relations):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random() > 0.5]}
    # for i in range(len(batch_placeholders)):
    #     feed_dict[batch_placeholders[i]] = batches[i]
    #     feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
    feed_dict[batch_placeholders] = batches
    for i in range(num_relations):
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))]
    return feed_dict

def accuracy(predictions, num_examples):
    # print("Beginning building accuracy")
    true_count = 0.
    for i in range(len(predictions[0])):
        if predictions[0][i] > 0:
            true_count += 1.0
    precision = float(true_count) / float(num_examples)

    return precision

def run_training():
    print("Begin!")
    # python list of (e1, R, e2) for entire training set in string form
    print("Load training data...")
    raw_training_data = ntn_input.load_training_data(params.data_path)
    print("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    # python list of (e1, R, e2) for entire training set in index form
    # subject, predicate, object
    indexed_training_data = data_to_indexed(raw_training_data, entities_list, relations_list)
    print("Load embeddings...")
    # wordvecs, ids
    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        print("Starting to build graph "+str(datetime.datetime.now()))
        # batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 4), name='batch_'+str(i)) for i in range(num_relations)]
        batch_placeholders = tf.placeholder(tf.int32, shape=(None, 4))
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_'+str(i)) for i in range(num_relations)]
        target_placeholder = tf.placeholder(tf.int32, shape=(None, 11))

        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) # Which of e1 or e2 to corrupt?
        # inference = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
        #         num_entities, num_relations, slice_size, batch_size, False, label_placeholders)
        logits, targets = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
                num_entities, num_relations, slice_size, batch_size, False, label_placeholders)
        # loss = ntn.loss(inference, params.regularization)
        loss = ntn.loss(logits, targets, params.regularization, num_relations, batch_size)
        eval_correct = ntn.eval(logits, targets)

        training = ntn.training(loss, params.learning_rate)
        # evaluate = ntn.eval(inference)

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.trainable_variables())
        for i in range(1, num_iters):
            print("Starting iter " + str(i) + " " + str(datetime.datetime.now()))
            # randomised subjects, predicates, objects, for given predicate
            data_batch = get_batch(batch_size, indexed_training_data, num_entities, corrupt_size)
            # relation, e1s, e2s, e_corrupts, targets
            relation_batches = split_batch(data_batch, num_relations)

            # if i % params.save_per_iter == 0:
            #     saver.save(sess, params.output_path + "/" + params.data_name + str(i) + '.sess')

            # print('indexed training sample:', relation_batches[0][3])
            feed_dict = fill_feed_dict(relation_batches, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder, num_relations)
            _, cost, acc = sess.run([training, loss, eval_correct], feed_dict=feed_dict)
            print('cost:', cost)

            # acc = accuracy(predictions, batch_size)
            print('acc:', acc)

         # Validation
         # logits, targets = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
         #        num_entities, num_relations, slice_size, batch_size, False, label_placeholders)
         # eval_correct = ntn.eval(logits, targets)
         # randomised subjects, predicates, objects, for given predicate
        data_batch = get_batch_val(batch_size, indexed_training_data, num_entities, corrupt_size)
        # relation, e1s, e2s, e_corrupts, targets
        relation_batches = split_batch(data_batch, num_relations)
        feed_dict = fill_feed_dict(relation_batches, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder, num_relations)

        cost_val, acc_val = sess.run([loss, eval_correct], feed_dict=feed_dict)
        print('cost_val:', cost_val)
        print('acc_val:', acc_val)

def main(argv):
    run_training()

if __name__=="__main__":
    tf.app.run()
