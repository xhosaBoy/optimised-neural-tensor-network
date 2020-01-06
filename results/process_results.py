# std
import os
import sys
import re
import csv
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(filename, dirname=None):
    root = os.path.dirname(os.path.dirname(__file__))
    logger.debug(f'root: {root}')

    path = os.path.join(root, dirname, filename) if dirname else os.path.join(root, filename)
    logger.debug(f'path: {path}')

    return path


def parse_results(path):
    with open(path, 'r') as resultsfile:
        pattern = re.compile(r'(epoch: [0-9]+), (([a-z]+)_[a-z]+: [0-9]+\.?[0-9]*)')
        results_cost = defaultdict(list)
        results_accuracy = defaultdict(list)

        for line in resultsfile:
            line = line.strip()
            logger.debug(f'line: {line}')
            record = pattern.findall(line)
            logger.debug(f'record: {record}')

            if record:
                record, = record
                metric = record[2]
                logger.debug(f'metric: {metric}')
                value = float(record[1].split(':')[1].strip())
                logger.debug(f'value: {value}')

                if metric == 'cost':
                    results_cost[record[0]].append(value)
                    logger.debug(f'results_cost: {results_cost}')
                elif metric == 'accuracy':
                    results_accuracy[record[0]].append(value)
                    logger.debug(f'results_accuracy: {results_accuracy}')

    logger.info(f'results_cost: {results_cost}')
    logger.info(f'results_accuracy: {results_accuracy}')

    return results_cost, results_accuracy


def write_results(results_cost_baseline,
                  results_accuracy_baseline,
                  results_cost_experiment,
                  results_accuracy_experiment):

    with open('rntn_train_validate_and_test_freebase_cost.csv', mode='w') as resultsfile:
        csv_writer = csv.writer(resultsfile)
        csv_writer.writerow(['cost_training_baseline',
                             'cost_validation_baseline',
                             'cost_test_baseline',
                             'cost_training_experiment',
                             'cost_validation_experiment',
                             'cost_test_experiment'])
        for epoch in results_cost_baseline:
            results_cost = results_cost_baseline[epoch]
            results_cost.extend(results_cost_experiment[epoch])
            logger.debug(f'results_cost: {results_cost}')
            csv_writer.writerow(results_cost)

    with open('rntn_train_validate_and_test_freebase_accuracy.csv', mode='w') as resultsfile:
        csv_writer = csv.writer(resultsfile)
        csv_writer.writerow(['accuracy_training_baseline',
                             'accuracy_validation_baseline',
                             'accuracy_test_baseline',
                             'accuracy_training_experiment',
                             'accuracy_validation_experiment',
                             'accuracy_test_experiment'])
        for epoch in results_accuracy_baseline:
            results_accuracy = results_accuracy_baseline[epoch]
            results_accuracy.extend(results_accuracy_experiment[epoch])
            logger.debug(f'results_accuracy: {results_accuracy}')
            csv_writer.writerow(results_accuracy)


def main():
    path_baseline = get_path('rntn_train_validate_and_test_freebase_baseline.log', 'results')
    path_experiment = get_path('rntn_train_validate_and_test_freebase_experiment.log', 'results')

    logger.info('Parsing results...')
    results_cost_baseline, results_accuracy_baseline = parse_results(path_baseline)
    logger.info('Parsing results complete!')

    logger.info('Parsing results...')
    results_cost_experiment, results_accuracy_experiment = parse_results(path_experiment)
    logger.info('Parsing results complete!')

    logger.info('Writing results...')
    write_results(results_cost_baseline,
                  results_accuracy_baseline,
                  results_cost_experiment,
                  results_accuracy_experiment)
    logger.info('Writing results complete!')


if __name__ == '__main__':
    logger.info('START!')
    main()
    logger.info('DONE!')
