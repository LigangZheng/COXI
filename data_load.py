import h5py
import warnings
import pickle as pk
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


# parts of code referred from https://github.com/ovshake/cobra/blob/master/dataloader.py
# parts of code referred from https://github.com/penghu-cs/DSCMR/blob/master/load_data.py

class CustomDataSet(Dataset):
	def __init__(self, images, texts, labels):
		self.images = images
		self.texts = texts
		self.labels = labels

	def __getitem__(self, index):
		img = self.images[index]
		text = self.texts[index]
		label = self.labels[index]
		return img, text, label

	def __len__(self):
		count = len(self.images)
		assert len(
			self.images) == len(self.labels)
		return count


def ind2vec(ind, N=None):
	ind = np.asarray(ind)
	if N is None:
		N = ind.max() + 1
	return np.arange(N) == np.repeat(ind, N, axis=1)

def load_data(args):
    if args.dataset == 'pascal':
        path = './data/pascal/'
        img_train = loadmat(path+"train_img.mat")['train_img']
        img_test = loadmat(path + "test_img.mat")['test_img']
        text_train = loadmat(path+"train_txt.mat")['train_txt']
        text_test = loadmat(path + "test_txt.mat")['test_txt']
        label_train = loadmat(path+"train_img_lab.mat")['train_img_lab']
        label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

        img_test, img_val = img_test, img_test
        text_test, text_val = text_test, text_test
        label_test, label_val = label_test, label_test

        label_train = ind2vec(label_train).astype(int)
        label_test = ind2vec(label_test).astype(int)
        label_val = ind2vec(label_val).astype(int)

        imgs = {'train': img_train, 'test': img_test, 'val': img_val}
        texts = {'train': text_train, 'test': text_test, 'val': text_val}
        labels = {'train': label_train, 'test': label_test, 'val': label_val}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train', 'test', 'val']}

    elif args.dataset == 'wiki':
        path = './data/wiki/' 
        image =loadmat(path + 'images.mat')['images']
        text = loadmat(path + 'texts.mat')['texts']
        label = loadmat(path + 'labels.mat')['labels'].squeeze()

        train_len = 2173
        test_len = 462 + train_len
        # val_len = test_len + 231

        img_train, img_test = image[:train_len].astype('float32'), image[train_len:test_len].astype('float32')
        img_val = image[test_len:].astype('float32')
        text_train, text_test = text[:train_len].astype('float32'), text[train_len:test_len].astype('float32')
        text_val = text[test_len:].astype('float32')
        label_train, label_test= label[:train_len], label[train_len:test_len]
        label_val = label[test_len:]

        label_train = np.eye(10)[label_train].astype(int)
        label_test = np.eye(10)[label_test].astype(int)
        label_val = np.eye(10)[label_val].astype(int)

        imgs = {'train': img_train, 'test': img_test, 'val': img_val}
        texts = {'train': text_train, 'test': text_test, 'val': text_val}
        labels = {'train': label_train, 'test': label_test, 'val': label_val}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train', 'test', 'val']}
    
    elif args.dataset == 'xmedia':
        xmedia = loadmat("./data/xmedia/Features.mat")

        img_train = xmedia['I_tr_CNN'].astype('float32')
        text_train = xmedia['T_tr_BOW'].astype('float32')
        label_train = xmedia['trImgCat'].squeeze()

        img_test = xmedia['I_te_CNN'].astype('float32')
        text_test = xmedia['T_te_BOW'].astype('float32')
        label_test = xmedia['teTxtCat'].squeeze()

        label_train = np.eye(20)[label_train-1].astype(int)
        label_test = np.eye(20)[label_test-1].astype(int)

        img_val, img_test = img_test[0:500], img_test[500:]
        text_val, text_test = text_test[0:500], text_test[500:]
        label_val, label_test = label_test[0:500], label_test[500:]

        imgs = {'train': img_train, 'test': img_test, 'val': img_test}
        texts = {'train': text_train, 'test': text_test, 'val': text_test}
        labels = {'train': label_train, 'test': label_test, 'val': label_test}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train', 'test', 'val']}

    elif args.dataset == 'nuswide':
        nuswide = loadmat("./data/nuswide/Features.mat")

        img_train = nuswide['I_tr_CNN'].astype('float32')
        text_train = nuswide['T_tr_BOW'].astype('float32')
        label_train = nuswide['trImgCat'].squeeze()

        img_test = nuswide['I_te_CNN'].astype('float32')
        text_test = nuswide['T_te_BOW'].astype('float32')
        label_test = nuswide['teTxtCat'].squeeze()

        label_train = np.eye(20)[label_train-1].astype(int)
        label_test = np.eye(20)[label_test-1].astype(int)

        img_val, img_test = img_test[0:4000], img_test[4000:]
        text_val, text_test = text_test[0:4000], text_test[4000:]
        label_val, label_test = label_test[0:4000], label_test[4000:]

        imgs = {'train': img_train, 'test': img_test, 'val': img_test}
        texts = {'train': text_train, 'test': text_test, 'val': text_test}
        labels = {'train': label_train, 'test': label_test, 'val': label_test}
        dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x])
                for x in ['train', 'test', 'val']}

    else:
        warnings.warn("data is no list")



    shuffle = {'train': True, 'test': False, 'val': True}

    dataloader = {x: DataLoader(dataset[x], batch_size=args.batch_size,
                                shuffle=shuffle[x], num_workers=0) for x in ['train', 'test', 'val']}

    img_dim = img_train.shape[1]
    text_dim = text_train.shape[1]
    num_class = label_train.shape[1]

    print(img_train.shape)
    print(img_test.shape)

    input_data_par = {}
    input_data_par['img_test'] = img_test
    input_data_par['text_test'] = text_test
    input_data_par['label_test'] = label_test
    input_data_par['img_train'] = img_train
    input_data_par['text_train'] = text_train
    input_data_par['label_train'] = label_train
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class
    return dataloader, input_data_par
