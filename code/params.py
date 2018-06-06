import math

data_number = 0 #0 - Wordnet, 1 - Freebase
data_name = 'Wordnet' if data_number == 0 else 'Freebase'

data_path = '../data/' + data_name
output_path = '../output/' + data_name+'/'

num_iter = 500
train_both = False
batch_size = 10000
# corrupt_size = 10 # how many negative examples are given for each positive example?
corrupt_size = 1 # how many negative examples are given for each positive example?
embedding_size = 100
slice_size = 3 # depth of tensor for each relation
regularization = 0.001 #parameter \lambda used in L2 normalization
in_tensor_keep_normal = False
save_per_iter = 10
learning_rate = 1

output_dir = ''

