import scipy.io

import numpy as np

def loading_data(path):
	image_path = path + "image.mat"
	label_path = path + "label.mat"
	tag_path = path + "tag.mat"

	images = scipy.io.loadmat(image_path)['Image']   # [13696, 224, 224, 3]
	tags = scipy.io.loadmat(tag_path)['Tag']     # [13696, 4945]
	labels = scipy.io.loadmat(label_path)["Label"]    # [13696, 32]

	# images = file['images'][:].transpose(0,3,2,1)
	# labels = file['LAll'][:].transpose(1,0)
	# tags = file['YAll'][:].transpose(1,0)

	return images, tags, labels


def split_data(images, tags, labels, QUERY_SIZE, TRAINING_SIZE, DATABASE_SIZE):

	X = {}
	index_all = np.random.permutation(QUERY_SIZE + DATABASE_SIZE)
	ind_Q = index_all[0:QUERY_SIZE]
	ind_T = index_all[QUERY_SIZE:TRAINING_SIZE + QUERY_SIZE]
	ind_R = index_all[QUERY_SIZE:DATABASE_SIZE + QUERY_SIZE]

	X['query'] = images[ind_Q, :, :, :]
	X['train'] = images[ind_T, :, :, :]
	X['retrieval'] = images[ind_R, :, :, :]

	Y = {}
	Y['query'] = tags[ind_Q, :]
	Y['train'] = tags[ind_T, :]
	Y['retrieval'] = tags[ind_R, :]

	L = {}
	L['query'] = labels[ind_Q, :]
	L['train'] = labels[ind_T, :]
	L['retrieval'] = labels[ind_R, :]
	return X, Y, L
