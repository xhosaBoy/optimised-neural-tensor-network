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
    fixtures, _ = os.path.split(os.path.dirname(__file__))
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


def get_records(entityfile):

    with open(entityfile, 'r') as entityfile:
        records = []

        for line in entityfile:
            record = {}
            intelligible_name = line.strip()
            logger.debug(f'intelligible_name: {intelligible_name}')

            pattern = re.compile(r'^__([0-9]*\b)')
            matches = pattern.search(intelligible_name)
            logger.debug(f'matches: {matches}')

            if matches:
                name, sense_index = matches.group(1), ''
            else:
                pattern = re.compile(r'^__([a-zA-Z0-9\'\._/-]*)(_[0-9]*)')
                matches, = pattern.findall(intelligible_name)
                logger.debug(f'matches: {matches}')
                name, sense_index = matches

            logger.debug(f'name: {name}, sense_index: {sense_index}')

            record['name'] = name
            record['sense_index'] = int(sense_index.replace('_', '')) if sense_index else None

            logger.debug(f'record: {record}')
            records.append(record)

        logger.info(f'number of records: {len(records)}')

    return records


def main():
    connection = get_connection('scientist',
                                '*********',
                                '127.0.0.1',
                                '5432',
                                'tensor_factorisation_wordnet')

    tablename = 'entity'
    entityfile = get_path('data/Wordnet', 'entities.txt')
    logger.debug(f'entityfile: {entityfile}')

    logger.info('Getting records...')
    records = get_records(entityfile)
    logger.info('Successfully got records!')

    logger.info('Inserting records...')
    insert_records(records, tablename, connection)
    logger.info('Successfully inserted records!')


if __name__ == '__main__':
    logger.info('Starting ETL...')
    main()
    logger.info('DONE!')
