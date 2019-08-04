# std
import os
import re

# 3rd party
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(2019)


def get_path(folder='results', filename='train_wordnet.log'):
    project = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    folder = folder
    filename = filename
    path = os.path.join(project, folder, filename)
    return path


def main():

    iteration = []
    training_cost = []
    training_accuracy = []
    validation_cost = []
    validation_accuracy = []

    path = get_path(filename='train_freebase.log')

    find_iteration = re.compile(r'ITERATION: ([0-9]+)')

    find_cost_train = re.compile(r'training cost: ([0-9]+\.[0-9]+)')
    find_accuracy_train = re.compile(r'training accuracy: ([0-9]+\.[0-9]+)')
    find_cost_validation = re.compile(r'validation cost: ([0-9]+\.[0-9]+)')
    find_accuracy_validation = re.compile(r'validation accuracy: ([0-9]+\.[0-9]+)')

    with open(path, 'r') as results:
        for line in results:
            iteration_training = find_iteration.findall(line)
            cost_training = find_cost_train.findall(line)
            accuracy_training = find_accuracy_train.findall(line)

            cost_validation = find_cost_validation.findall(line)
            accuracy_validation = find_accuracy_validation.findall(line)

            if iteration_training:
                iteration.append(int(iteration_training[0]))
            if cost_training:
                training_cost.append(float(cost_training[0]))
            if accuracy_training:
                training_accuracy.append(float(accuracy_training[0]))
            if cost_validation:
                validation_cost.append(float(cost_validation[0]))
            if accuracy_validation:
                validation_accuracy.append(float(accuracy_validation[0]))

    print(f'iteration: {iteration}')
    print(f'training cost: {training_cost}')
    print(f'training accuracy: {training_accuracy}')
    print(f'validation cost: {validation_cost}')
    print(f'validation accuracy: {validation_accuracy}')

    print(f'iteration: {len(iteration)}')
    print(f'training_accuracy: {len(training_accuracy)}')

    x1_training = np.linspace(0, 320, 320)
    x2_training = np.linspace(0, 320, 320)
    x1_validation = np.linspace(0, 320, 10)
    x2_validation = np.linspace(0, 320, 10)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(x1_training, training_cost, label='training cost')
    ax2.plot(x2_training, training_accuracy, label='training accuracy')
    ax1.plot(x1_validation, validation_cost, label='validation cost')
    ax2.plot(x2_validation, validation_accuracy, label='validation accuracy')

    ax1.set_title('Training and Validation Cost: Freebase')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('')
    ax1.legend()
    ax2.set_title('Training and Validation Accuracy: Freebase')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Quantile')
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main()
