import numpy as np
from PIL import Image
import os
import pdb

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class NFOV(data.Dataset):

    def __init__(self, root, vocab, video_len, MAX_LENGTH, train=True):
        self.root = root
        self.train = train
        self.vocab = vocab
        self.video_len = video_len
        self.max_sub_len = MAX_LENGTH
        self.train_narrative = 'data/subtitle_train_onlyword/'
        self.test_narrative = 'data/subtitle_test_onlyword/'
        self.dictionary_path = 'data/dictionary.npy'
        self.glove300 = np.load('data/glove300.npy').item()
        self.glove_dim = 300
        self.NFOV_x = 1280
        self.transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406),
                                              (0.229, 0.224, 0.225))])

        if self.train:
            self.training_data = np.sort(np.array(open('data/train_data.txt').read().split('\n')))[1:]
            self.max_img_len = self.get_video_length(percentage = 90)
        else:
            self.testing_data = np.sort(np.array(open('data/test_data.txt').read().split('\n')))[1:]
            self.max_img_len = self.get_video_length(percentage = 100)


    def __getitem__(self, index, augment=False):

        if self.train:
            ''' initialize '''
            img_out_all = []
            n_words_out = np.zeros((self.video_len, self.max_sub_len))
            wordvec = np.zeros((self.video_len, self.max_sub_len, self.glove_dim))

            img = self.training_data[index]
            frames = len(os.listdir(self.root+img))
            if frames <= self.max_img_len:

                ''' load narratives '''
                narrative = open(self.train_narrative + img + '.txt').read().split('\n')[:-1]

                ''' random choose video clip '''
                if frames > self.video_len:
                    rand_sample_video_start = np.random.randint(frames - self.video_len)
                    range_idx = range(rand_sample_video_start, rand_sample_video_start + self.video_len)
                else:
                    rand_sample_video_start = 0
                    range_idx = range(frames)

                for edx, ele in enumerate(range_idx):
                    ''' extract pretrained word vectors '''
                    tokens = narrative[ele].split(' ')[:-1]
                    caption_tmp = []
                    wordvec_tmp = []
                    caption_tmp.append(self.vocab('<start>'))
                    wordvec_tmp.append(self.glove300['<start>'])
                    for token in tokens:
                        caption_tmp.append(self.vocab(token))
                        if token in self.glove300:
                            wordvec_tmp.append(self.glove300[token])
                        else:
                            wordvec_tmp.append(self.glove300['<unk>'])
                    caption_tmp.append(self.vocab('<end>'))
                    wordvec_tmp.append(self.glove300['<end>'])
                    n_words_out[edx, :len(tokens)+2] = np.array(caption_tmp)
                    wordvec[edx, :len(tokens)+2] = np.array(wordvec_tmp)

                    ''' load images '''
                    img_name = self.root + img + '/image_' + str("%06d" %int((ele*30)+1)) + '.jpg'
                    image = np.array(Image.open(img_name).convert('RGB'))[:,0:self.NFOV_x,:]

                    ''' data augmentation '''
                    if augment == True:
                        rand_start = np.random.randint(0,self.NFOV_x)
                        image = np.column_stack((image[:,rand_start:,:],image[:,:rand_start,:]))
                    image = self.transform(image)
                    img_out_all.append(image)

                if len(narrative) < self.video_len:
                    for fff in range(self.video_len-len(narrative)):
                        img_out_all.append(torch.zeros((image.shape)))

            order_out = torch.FloatTensor(wordvec)
            words_out = torch.LongTensor(n_words_out)
            image_out = torch.stack(img_out_all ,dim = 0)
            index_out = torch.LongTensor(([index]))

        else:
            ''' initialize '''
            img_out_all = []
            n_words_out = np.zeros((self.max_img_len, self.max_sub_len))
            wordvec = np.zeros((self.max_img_len, self.max_sub_len, self.glove_dim))

            img = self.testing_data[index]
            frames = len(os.listdir(self.root+img))
            if frames <= self.max_img_len:

                ''' load narratives '''
                narrative = open(self.test_narrative + img + '.txt').read().split('\n')[:-1]

                for edx, ele in enumerate(range(len(narrative))):
                    ''' extract pretrained word vectors '''
                    tokens = narrative[ele].split(' ')[:-1]
                    caption_tmp = []
                    wordvec_tmp = []
                    caption_tmp.append(self.vocab('<start>'))
                    wordvec_tmp.append(self.glove300['<start>'])
                    for token in tokens:
                        caption_tmp.append(self.vocab(token))
                        if token in self.glove300:
                            wordvec_tmp.append(self.glove300[token])
                        else:
                            wordvec_tmp.append(self.glove300['<unk>'])
                    caption_tmp.append(self.vocab('<end>'))
                    wordvec_tmp.append(self.glove300['<end>'])
                    n_words_out[edx, :len(tokens)+2] = np.array(caption_tmp)
                    wordvec[edx, :len(tokens)+2] = np.array(wordvec_tmp)

                    ''' load images '''
                    img_name = self.root + img + '/image_' + str("%06d" %int((ele*30)+1)) + '.jpg'
                    image = np.array(Image.open(img_name).convert('RGB'))[:,0:self.NFOV_x,:] 
                    image = self.transform(image)
                    img_out_all.append(image)
                if len(img_out_all) < self.max_img_len:
                    for fff in range(self.max_img_len-len(img_out_all)):
                        img_out_all.append(torch.zeros((image.shape)))

            image_out = torch.stack(img_out_all ,dim = 0)
            order_out = torch.FloatTensor(wordvec)
            words_out = torch.LongTensor(n_words_out)
            index_out = torch.LongTensor(([index]))

        return image_out, order_out, words_out, index_out
    
    def __len__(self):
        if self.train: return len(self.training_data)
        else: return len(self.testing_data)

    def get_video_length(self, percentage = 100):
        length = []
        for ii, ele in enumerate(os.listdir(self.root)):
            length.append(len(os.listdir(self.root+ele)))
        video_length = np.percentile(np.array(length), percentage)
        return int(video_length)

    def get_dictionary_size(self):
        return np.load(self.dictionary_path).shape[0]
