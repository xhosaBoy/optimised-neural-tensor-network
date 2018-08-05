import os
from pprint import pprint

dirname = os.path.split(os.path.dirname(__file__))[0]
filename = 'data/Wordnet/testIII.txt'
path = os.path.join(dirname, filename)
print(path)

with open(path, 'r') as fhand:
  count = 0
  entity_list = list()
  relation_list = list()
  for line in fhand:
    # print(line)
    entity1 = line.split()[0]
    entity2 = line.split()[2]
    relation = line.split()[1]
    entity_list.extend((entity1, entity2))
    relation_list.append(relation)
    if entity1 == '__deed_2':
      # entity_list.append(entity)
      print(line)
    count += 1
  print(count)

entity_set = set(entity_list)
print(len(entity_set))
pprint(entity_set)

relation_set = set(relation_list)
print(len(relation_set))
pprint(relation_set)

filename_entity = 'data/Wordnet/entities_test.txt'
path_entity = os.path.join(dirname, filename_entity)
filename_relation = 'data/Wordnet/relations_test.txt'
path_relation = os.path.join(dirname, filename_relation)


with open(path_entity, 'a') as fhand:
  for entity in entity_set:
    fhand.write(str(entity))
    fhand.write('\n')

# with open(path_relation, 'a') as fhand:
#   for relation in relation_set:
#     fhand.write(str(relation))
#     fhand.write('\n')
