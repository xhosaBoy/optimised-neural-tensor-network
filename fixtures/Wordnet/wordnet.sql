DROP TABLE if EXISTS entity;
DROP TABLE if EXISTS relation;
DROP TABLE if EXISTS train;
DROP TABLE if EXISTS valid;
DROP TABLE if EXISTS test;

CREATE TABLE entity(
   synset_id SERIAL PRIMARY KEY,
   name text,
   sense_index integer,
   UNIQUE (name, sense_index)
);

CREATE TABLE relation(
   id SERIAL PRIMARY KEY,
   name text UNIQUE
);

CREATE TABLE train(
   id SERIAL PRIMARY KEY,
   subject text,
   subject_sense_index integer,
   predicate text,
   object text,
   object_sense_index integer,
   UNIQUE (subject, subject_sense_index, predicate, object, object_sense_index)
);

CREATE TABLE valid(
   id SERIAL PRIMARY KEY,
   subject text,
   subject_sense_index integer,
   predicate text,
   object text,
   object_sense_index integer,
   UNIQUE (subject, subject_sense_index, predicate, object, object_sense_index)
);

CREATE TABLE test(
   id SERIAL PRIMARY KEY,
   subject text,
   subject_sense_index integer,
   predicate text,
   object text,
   object_sense_index integer,
   UNIQUE (subject, subject_sense_index, predicate, object, object_sense_index)
);