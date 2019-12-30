# std
import os
import sys
import re
import logging

# 3rd party
import psycopg2
from psycopg2.extras import Json
from psycopg2.extensions import AsIs


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def get_path(dirname, filename=None):
    """File path getter.
        Args:
            dirname (str): File directory.
            filename (str): File name.
        Returns:
            filepath (str): Full file file path.
        """
    fixtures, _ =  os.path.split(os.path.dirname(__file__))
    project = os.path.dirname(fixtures)
    path = os.path.join(project, dirname, filename) if filename else os.path.join(project, dirname)

    return path


def get_connection(user, password, host, port, database):
    """Database connection getter.
        Args:
            user (str): database user.
            password (str): database user password.
            host (str): database IP.
            port (str): database port.
            database (str): database name.
        Returns:
            connection (obj): postgres database connection.
        """
    connection = psycopg2.connect(user=user,
                                  password=password,
                                  host=host,
                                  port=port,
                                  database=database)
    return connection


def insert_record(record, tablename, cursor, connection):

    columns = record.keys()
    logger.debug(f'columns: {columns}')
    values = record.values()
    logger.debug(f'values: {values}')
    values = list(map(lambda x: Json(x) if isinstance(x, dict) else x, values))

    insert_statement = 'INSERT INTO %s (%s) VALUES %s'
    logger.debug(f"cursor.mogrify: {cursor.mogrify(insert_statement, (AsIs(tablename), AsIs(','.join(columns)), tuple(values)))}")

    try:
        cursor.execute(insert_statement, (AsIs(tablename), AsIs(','.join(columns)), tuple(values)))
    except Exception as e:
        logger.error(f'Could not insert into {tablename}, {e}')

    connection.commit()
    count = cursor.rowcount
    logger.debug(f'{count} Record inserted successfully into {tablename} table')


def insert_records(records, tablename, connection):

    with connection as con:
        cursor = con.cursor()

        for record in records:
            insert_record(record, tablename, cursor, con)


def parse_entity(entity):
    pattern = re.compile(r'([a-zA-Z0-9_-]*)')
    matches = pattern.findall(entity)
    logger.debug(f'matches: {matches}')

    entity_name, _ = matches
    logger.debug(f'entity_name: {entity_name}')

    return entity_name


def parse_record(filename, line):
    record = {}

    if filename == 'train.txt':
        subject, predicate, obj = line.strip().split('\t')
    else:
        subject, predicate, obj, _ = line.strip().split('\t')
    logger.debug(f'subject: {subject}, predicate: {predicate}, object: {obj}')

    predicate = predicate.replace('_', ' ').strip()
    logger.debug(f'predicate: {predicate}')

    logger.debug(f'Parsing subject...')
    subject = parse_entity(subject)
    logger.debug(f'Parsing subject complete!')
    logger.debug(f'Parsing object...')
    obj = parse_entity(obj)
    logger.debug(f'Parsing object complete!')

    logger.debug(f'obj: {obj}')

    record['subject'] = subject
    record['predicate'] = predicate
    record['object'] = obj
    logger.debug(f'record: {record}')

    return record


def get_records(tripletfile):

    with open(tripletfile, 'r') as factfile:
        records = []

        for line in factfile:
            logger.debug(f'line: {line.strip()}')
            _, filename = os.path.split(tripletfile)
            logger.debug(f'filename: {filename}')

            logger.debug(f'Parsing record...')
            record = parse_record(filename, line)
            logger.debug(f'Parsing complete!')
            records.append(record)

        logger.info(f"number of {filename} records: {len(records)}")

    return records


def main():
    logger.info('Connecting to database...')
    connection = get_connection('scientist',
                                '*********',
                                '127.0.0.1',
                                '5432',
                                'tensor_factorisation_freebase')
    logger.info('Connecting to database successful!')

    tripletfile = get_path('data/Freebase')
    logger.debug(f'tripletfile: {tripletfile}')

    dirname, = list(os.walk(tripletfile))
    _, _, filenames = dirname
    experiment = ['train.txt', 'dev.txt', 'test.txt']

    for filename in filenames:
        if filename in experiment:
            if filename == 'dev.txt':
                tablename = 'valid'
            else:
                tablename, _ = filename.split('.')

            logger.debug(f'tablename: {tablename}')
            filename = get_path('data/Freebase', filename)
            logger.debug(f'filename: {filename}')

            logger.info(f'Getting {tablename} records...')
            records = get_records(filename)
            logger.debug(f'records: {records}')
            logger.debug(f'number of records: {len(records)}')
            logger.info('Completed getting records!')

            logger.info('Inserting records...')
            insert_records(records, tablename, connection)
            logger.info('Completed getting records!')


if __name__ == '__main__':
    logger.info('Starting ETL...')
    main()
    logger.info('DONE!')
