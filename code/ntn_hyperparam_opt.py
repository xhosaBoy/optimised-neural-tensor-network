# std
import pickle

# 3rd party
import numpy as np
import tensorflow as tf

# internal
import ntn_input
import ntn
import params
from ntn_train import run_training
from ntn_plot import plot_multi
from ntn_eval import data_to_indexed, data_to_relation_sets, do_eval


def sample_log_scale(v_min=1e-6, v_max=1.):
    '''Sample uniformly on a log-scale from 10**v_min to 10**v_max.'''
    return np.exp(np.random.uniform(np.log(v_min), np.log(v_max)))


def sample_model_architecture_and_hyperparams(max_depth=10,
                                              max_corrupt=10,
                                              lr_min=1e-3,
                                              lr_max=1e-1,
                                              mom_min=0.5,
                                              mom_max=1.,
                                              l2_min=1e-4,
                                              l2_max=1.):
    '''Generate a random model architecture & hyperparameters.'''

    # Sample the architecture.
    slice_size = np.random.choice(range(1, max_depth + 1))
    corrupt_size = np.random.choice(range(1, max_corrupt + 1))

    # Sample the training parameters.
    l2_lambda = sample_log_scale(l2_min, l2_max)
    lr = sample_log_scale(lr_min, lr_max)
    mom_coeff = sample_log_scale(mom_min, mom_max)

    # Build base model definitions:
    model_params = {
        'slice_size': 2}

    # Specify the training hyperparameters:
    training_params = {
        'batch_size': 10000,
        'corrupt_size': 1,
        'lr': 0.095816854173317881,
        'l2_lambda': 0.045865217294355581,
        'mom_coeff': 0.51650816560051749,
        'optimizer_fn': tf.train.AdamOptimizer(learning_rate=lr),
        'val_per_iter': 10,
        'stop_early': True,
        'num_epochs': 500}

    return model_params, training_params


def build_train_eval_and_plot(build_params, train_params, verbose=True):
    train_costs, train_accs, val_costs, val_accs = run_training(**build_params, **train_params)
    return train_costs, train_accs, val_costs, val_accs


def build_optimal_model():
    # Perform a random search over hyper-parameter space this many times.
    results = []
    NUM_EXPERIMENTS = 1

    for i in range(NUM_EXPERIMENTS):

        # Sample the model and hyperparams we are using.
        model_params, training_params = sample_model_architecture_and_hyperparams()

        print("RUN: %d out of %d:" % (i + 1, NUM_EXPERIMENTS))
        print("Sampled Architecture: \n", model_params)
        print("Hyper-parameters:\n", training_params)
        print()

        # Build, train, evaluate
        train_losses, train_accs, val_losses, val_accs = build_train_eval_and_plot(
            build_params=model_params, train_params=training_params, verbose=False)

        ret = {'cost_training': train_losses, 'accuracy_training': train_accs,
               'cost_validation': val_losses, 'accuracy_validation': val_accs}

        with open('results_train_val.pkl', 'wb') as fhand:
            pickle.dump(ret, fhand)

        results.append((val_accs[-1], model_params, training_params))

        plot_multi([train_losses, val_losses], ['train', 'val'], 'loss', 'epoch',
                   [train_accs, val_accs], ['train', 'val'], 'accuracy', 'epoch')

    results.sort(key=lambda x: x[0], reverse=True)

    # Save results
    with open('results_plot.txt', 'w') as fhand:
        for r in results:
            print(r)
            fhand.write(str(r) + '\n')


def test_model():

    print("Beginning building testing")
    print(params.output_path)
    checkpoint = tf.train.latest_checkpoint(params.output_path, 'checkpoint')
    print(checkpoint)

    print("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    print("Load validation data...")
    test_data = ntn_input.load_test_data(params.data_path)
    test_data = data_to_indexed(test_data, entities_list, relations_list)

    batch_size = len(test_data)
    num_entities = len(entities_list)
    num_relations = len(relations_list)

    slice_size = 2

    init_word_embeds, entity_to_wordvec = ntn_input.load_init_embeds(params.data_path)
    batches, labels = data_to_relation_sets(test_data, num_relations)

    with tf.Graph().as_default():

        sess = tf.Session()
        batch_placeholders = [tf.placeholder(tf.float32, shape=(None, 3))
                              for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1))
                              for i in range(num_relations)]
        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1))

        _, inference = ntn.inference(batch_placeholders,
                                     corrupt_placeholder,
                                     init_word_embeds,
                                     entity_to_wordvec,
                                     num_entities,
                                     num_relations,
                                     slice_size,
                                     batch_size,
                                     True,
                                     label_placeholders)

        eval_correct = ntn.eval(inference)
        saver = tf.train.Saver()

        saver.restore(sess, checkpoint)

        test_accuracy = do_eval(sess,
                                eval_correct,
                                batch_placeholders,
                                label_placeholders,
                                corrupt_placeholder,
                                batches,
                                labels,
                                batch_size)

        with open('results_test.txt', 'w') as fhand:
            print('test accuracy:', test_accuracy)
            fhand.write(str(test_accuracy) + '\n')


def main():
    build_optimal_model()
    test_model()
    print('Done!')


if __name__ == "__main__":
    main()
