import math

data_number = 1 #1 - Wordnet, 0 - Freebase
data_name = 'Wordnet' if data_number else 'Freebase'

data_path = '../data/' + data_name
output_path = '../output/' + data_name + '/'

num_iter = 10
train_both = False
batch_size = 20000
# corrupt_size = 10 # how many negative examples are given for each positive example?
corrupt_size = 1 # how many negative examples are given for each positive example?
embedding_size = 100
slice_size = 3 # depth of tensor for each relation
regularization = 0.0001 #parameter \lambda used in L2 normalization
in_tensor_keep_normal = False
save_per_iter = 10
learning_rate = 0.01

output_dir = ''

