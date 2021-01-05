import numpy as np
import scipy.io

from load_data import loading_data
from load_data import split_data


# environmental setting: setting the following parameters based on your experimental environment.
select_gpu = '0'
per_process_gpu_memory_fraction = 1

MODEL_DIR = '/home/ubuntu/chan/SHDCH-pytorch/DataSet/imagenet-vgg-f.mat'

DATA_DIR = '/home/ubuntu/chan/SHDCH-pytorch/DataSet/FashionVC/'
TRAINING_SIZE = 16862
QUERY_SIZE = 3000
DATABASE_SIZE = 16862
num_class1 = 8
num_class2 = 27
num_class = num_class1 + num_class2

# DATA_DIR = '/home/ubuntu/chan/SHDCH-pytorch/DataSet/Ssense/'
# TRAINING_SIZE = 13696
# query_size = 2000
# DATABASE_SIZE = 13696
# num_class1 = 4
# num_class2 = 28
# num_class = num_class1 + num_class2

phase = 'train'
checkpoint_dir = './checkpoint'
Savecode = './Savecode'
dataset_dir = 'Flickr'
netStr = 'alex'

SEMANTIC_EMBED = 512
MAX_ITER = 100
batch_size = 128
image_size = 224


images, tags, labels = loading_data(DATA_DIR)
dimTxt = tags.shape[1]
dimLab = labels.shape[1]

VERIFICATION_SIZE = 1000

X, Y, L = split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE)
train_L = L['train']
train_x = X['train']
train_y = Y['train']

query_L = L['query']
query_x = X['query']
query_y = Y['query']

retrieval_L = L['retrieval']
retrieval_x = X['retrieval']
retrieval_y = Y['retrieval']

num_train = train_x.shape[0]
numClass = train_L.shape[1]
dimText = train_y.shape[1]

Sim = (np.dot(train_L, train_L.transpose()) > 0).astype(int)*0.999

data = scipy.io.loadmat(MODEL_DIR)
mean = data['normalization'][0][0][0]

Epoch = 500
k_lab_net = 10
k_img_net = 15
k_txt_net = 15
k_dis_net = 1
save_freq = 1

bit = 16    # bit

alpha = 1
gamma = 1
beta = 1
eta = 1
delta = 1

# Learning rate
lr_lab = [np.power(0.1, x) for x in np.arange(2.0, MAX_ITER, 0.5)]
lr_img = [np.power(0.1, x) for x in np.arange(4.5, MAX_ITER, 0.5)]
lr_txt = [np.power(0.1, x) for x in np.arange(3.5, MAX_ITER, 0.5)]
lr_dis = [np.power(0.1, x) for x in np.arange(3.0, MAX_ITER, 0.5)]
