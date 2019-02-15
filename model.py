import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from resnet import resnet101
import VideoRecorder

class NFOV_gorund(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_length, batch_size):
        super(NFOV_gorund, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length
        self.NFOV_x = 480
        self.NFOV_y = 360
        self.img_hidden = 1024
        self.pano_axis = np.array([(x, y) for x in (np.array(range(-180,180,30))*np.pi/180)
                    for y in (np.array([-30, -15, 0, 15, 30])*np.pi/180)])
        
        ''' encode img '''
        self.CNN = resnet101(pretrained=True)
        self.attn_img = nn.Linear(self.img_hidden, self.hidden_size)
        self.attn_lang = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.encoded_att = nn.Linear(self.img_hidden, self.output_size)
        self.tanh = nn.Tanh()

    def forward(self, batch_size, img, lang_rep):
       
        imgWW = img.size()[3]
        encode_img_CNN = self.CNN(img)

        sphereW = encode_img_CNN.size(3)
        sphereH = encode_img_CNN.size(2)

        ''' extract NFoV '''
        warp_img = VideoRecorder.ImageRecorder(sphereW, sphereH, imgW=self.NFOV_x/(imgWW/sphereW))
        img_pano = []
        for ele, axis in enumerate(self.pano_axis):
            Px,Py = warp_img._sample_points(axis[0],axis[1])
            Px[np.where(np.greater_equal(Px,encode_img_CNN.size(-1)))] = 0
            img_pano.append(encode_img_CNN[:,:, Py.astype(int), Px.astype(int)].unsqueeze(1))
        img_pano = torch.cat(img_pano, dim=1)
        img_pano = img_pano.view(batch_size * len(self.pano_axis), *img_pano.size()[2:])
        encode_img = self.avgpool(img_pano).squeeze(-1).squeeze(-1)
        
        ''' attention mechanism '''
        attn_img_emb = self.attn_img(encode_img).view(batch_size, len(self.pano_axis), -1)
        attn_lang_emb = self.attn_lang(lang_rep).unsqueeze(1).repeat(1,len(self.pano_axis),1)
        attn_scores = self.attn_combine(self.tanh(attn_img_emb + attn_lang_emb)).squeeze(-1)

        ''' weighted sum '''
        attn_weights = self.softmax(attn_scores)
        attn_applied = attn_weights.unsqueeze(1).bmm(encode_img.view(batch_size, len(self.pano_axis), -1)).squeeze(1)
        attn_applied_encoded = self.tanh(self.encoded_att(attn_applied))

        attn_weights_neg = self.softmax(1 - attn_weights)
        attn_applied_neg = attn_weights_neg.unsqueeze(1).bmm(encode_img.view(batch_size, len(self.pano_axis), -1)).squeeze(1)
        attn_applied_neg_encoded = self.tanh(self.encoded_att(attn_applied_neg))

        return attn_applied_encoded, attn_applied_neg_encoded, attn_weights

    def initHidden(self, batch_size, use_cuda):
        attn_applied = Variable(torch.zeros(batch_size, self.img_hidden))
        if use_cuda:
            return attn_applied.cuda()
        else:
            return attn_applied

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, n_layers):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.GRU = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, batch_size, input, hidden):
        output_list = []
        for idx in range(input.size()[1]):
            for i in range(self.n_layers):
                hidden = self.GRU(input[:,idx], hidden)
            output_list.append(hidden)
        return output_list

    def initHidden(self, batch_size, use_cuda):
        hidden = Variable(torch.zeros(batch_size, self.hidden_size))
        memory = Variable(torch.zeros(batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda(), memory.cuda()
        else:
            return hidden, memory
