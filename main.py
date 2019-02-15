import numpy as np
import os, argparse, time, pickle
import heapq

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import model as model
from model_cap import DecoderRNN
from data_loader import NFOV
from evaluate import evaluate
from build_vocab import Vocabulary
from util import asMinutes, timeSince

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

def train(batch_size, epoch, encoder, VGM, decoder, optimizer, criterion, MAX_LENGTH, video_len):

    loss_epoch = []
    loss_pos_epoch = []
    loss_neg_epoch = []
    for batch_idx, (img, narrative, narrative_index, index) in enumerate(train_loader):
        print(index)
        ''' load data '''
        if use_cuda:
            img, narrative, narrative_index = img.cuda(), narrative.cuda(), narrative_index.cuda()
        img, narrative, narrative_index = Variable(img), Variable(narrative), Variable(narrative_index)

        loss_frame = 0
        loss_pos_frame = 0
        loss_neg_frame = 0
        for vi in range(video_len):
            loss_pos = 0
            loss_neg = 0
            
            ''' encode narratives '''
            encoder_hidden, encoder_memory = encoder.initHidden(batch_size, use_cuda)
            encoder_output_list = encoder(batch_size, narrative[:,vi,:], encoder_hidden)
            encoder_output = Variable(torch.FloatTensor(encoder_output_list[-1].size()).zero_()).cuda()
            end_word = np.zeros((batch_size))
            for bi in range(batch_size):
                if len((narrative_index[bi][vi] == 0).nonzero()) == MAX_LENGTH:
                    end_word[bi] = 0
                elif len((narrative_index[bi][vi] == 0).nonzero()) == 0:
                    end_word[bi] = MAX_LENGTH-1
                else:
                    end_word[bi] = (narrative_index[bi][vi] == 2).nonzero().squeeze(1)[0]
                encoder_output[bi, :] = encoder_output_list[int(end_word[bi])][bi,:]
            
            ''' VGM model '''
            NFOV_output, NFOV_output_neg, NFOV_att = my_model(batch_size, img[:,vi], encoder_output)

            ''' decode narratives '''
            decode_output = decoder.sample(NFOV_output, MAX_LENGTH)
            decode_output_neg = decoder.sample(NFOV_output_neg, MAX_LENGTH)
            decode_out_word = Variable(torch.FloatTensor(batch_size, len(decode_output), len(vocab)).zero_()).cuda()
            decode_out_word_neg = Variable(torch.FloatTensor(batch_size, len(decode_output), len(vocab)).zero_()).cuda()
            for bbb in range(batch_size):
                for eee in range(len(decode_output)):
                    decode_out_word[bbb,eee,:] = decode_output[eee][bbb]
                    decode_out_word_neg[bbb,eee,:] = decode_output_neg[eee][bbb]

            ''' calculate loss '''
            loss_batch = 0
            loss_batch_neg = 0
            for bbb in range(batch_size):
                if int(end_word[bbb]) > 0:
                    loss_pos += criterion(decode_out_word[bbb][:int(end_word[bbb]+1)], narrative_index[bbb,vi,:int(end_word[bbb]+1)])
                    loss_neg += criterion(decode_out_word_neg[bbb][:int(end_word[bbb]+1)], narrative_index[bbb,vi,:int(end_word[bbb]+1)])
            loss = 0.8*loss_pos - 0.2*torch.log(loss_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_frame += loss.data[0]
            loss_pos_frame += loss_pos.data[0]
            loss_neg_frame += loss_neg.data[0]

        loss_frame /= video_len
        loss_pos_frame /= video_len
        loss_neg_frame /= video_len
    
        loss_epoch.append(loss_frame)
        loss_pos_epoch.append(loss_pos_frame)
        loss_neg_epoch.append(loss_neg_frame)

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\
                Loss: {:.6f}\tLoss_pos: {:.6f}\tLoss_neg: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss_frame, loss_pos_frame, loss_neg_frame))
        if batch_idx == len(train_loader) - 2:
            break

    return np.mean(loss_epoch), np.mean(loss_pos_epoch), np.mean(loss_neg_epoch)

def test(batch_size, epoch, encoder, my_model, MAX_LENGTH):
    
    recall_vi = []
    precision_vi = []
    recall_all = []
    precision_all = []
    for batch_idx, (img, narrative, narrative_index, index) in enumerate(test_loader):
        img_length = img.size()[1]
        view_axis = np.zeros((batch_size, img_length, 2))
        view_axis_5 = np.zeros((batch_size, img_length, 5, 2))
        probability = np.zeros((batch_size, img_length, 5))

        ''' load data '''
        if use_cuda:
            img, narrative, narrative_index = img.cuda(), narrative.cuda(), narrative_index.cuda()
        img, narrative, narrative_index = Variable(img), Variable(narrative), Variable(narrative_index)

        for vi in range(img_length):
            recall = 0
            
            ''' encode narratives '''
            encoder_hidden, _ = encoder.initHidden(batch_size, use_cuda)
            encoder_output_list = encoder(batch_size, narrative[:,vi,:], encoder_hidden)
            encoder_output = Variable(torch.FloatTensor(encoder_output_list[-1].size()).zero_()).cuda()
            end_word = np.zeros((batch_size))
            for bi in range(batch_size):
                if len((narrative_index[bi][vi] == 0).nonzero()) == MAX_LENGTH:
                    end_word[bi] = 0
                elif len((narrative_index[bi][vi] == 0).nonzero()) == 0:
                    end_word[bi] = MAX_LENGTH-1
                else:
                    end_word[bi] = (narrative_index[bi][vi] == 2).nonzero().squeeze(1)[0]
                encoder_output[bi, :] = encoder_output_list[int(end_word[bi])][bi,:]
            
            ''' VGM model '''
            _, _, NFOV_att = my_model(batch_size, img[:,vi].detach(), encoder_output.detach())

            ''' extract top 1 and top 5 '''
            NFOV_att_detach = NFOV_att.detach().data.cpu().numpy()
            NFOV_att_max = np.argmax(NFOV_att_detach, axis=1)
            pano_axis = my_model.pano_axis.reshape(60,-1)
            view_axis[:,vi,:] = pano_axis[NFOV_att_max]
            NFOV_att_max5 = np.zeros((batch_size, 5))
            for bbb in range(batch_size):
                NFOV_att_max5[bbb] = heapq.nlargest(5,range(len(NFOV_att_detach[bbb])), NFOV_att_detach[bbb].take)
                for idx in range(len(NFOV_att_max5[0])):
                    probability[bbb,vi,idx] = NFOV_att_detach[bbb,int(NFOV_att_max5[bbb,idx])]
                    view_axis_5[bbb,vi,idx,:] = pano_axis[int(NFOV_att_max5[bbb,idx])]
        #np.savez("/home/Han/Han_NIPS/model/output/FeaMap2/End2Endloss91Ori/test_out_" + str(batch_idx), pano_view = view_axis_5, probability_5 = probability, video_index = index[:,0].numpy())

        ''' evaluate '''
        recall, idx_batch, precision = evaluate(view_axis, index.numpy()[0], img.size()[-2:], 0.5)
        recall_all.append(recall)
        precision_all.append(precision)
        recall_vi.append(np.mean(recall))
        precision_vi.append(np.mean(precision))
        print("Average recall at batch %d (file %d): %.3f" %(batch_idx, index.numpy()[0], np.mean(recall)))
        print("Average precision at batch %d (file %d): %.3f" %(batch_idx, index.numpy()[0], np.mean(precision)))
    return np.mean(recall_vi), recall_all, np.mean(precision_vi), precision_all

def trainEpoches(encoder, VGM, decoder, MAX_LENGTH, batch_size, n_epoches, video_len, print_every=1, plot_every=100, learning_rate=0.01):
    start = time.time()

    params = list(encoder.parameters()) + list(VGM.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr = learning_rate)
    criterion = nn.CrossEntropyLoss()

    for Epoches in range(1, n_epoches + 1):
        if int(Epoches)%20 == 1:
            learning_rate = learning_rate/10.
            optimizer = optim.Adam(params, lr = learning_rate)
        loss_epoch, loss_pos_epoch, loss_neg_epoch = train(batch_size, Epoches, encoder, VGM, decoder, optimizer, criterion, MAX_LENGTH, video_len)

        if Epoches % print_every == 0:
            print('Epoch %d is done.' % (Epoches))
            print('Overall time cose: %s (%d %d%%),\
                   Avg. Loss: %.4f,\
                   Avg. Pos Loss: %.4f,\
                   Avg. Neg Loss: %.4f' 
                   % (timeSince(start, float(Epoches) / n_epoches), Epoches, Epoches / n_epoches * 100, 
                   loss_epoch, loss_pos_epoch, loss_neg_epoch))
        
        if Epoches % 10 == 0:
            torch.save(encoder, args.save_dir + 'encoder_%d' % (Epoches))
            torch.save(my_model, args.save_dir + 'VGM_%d' % (Epoches))
            torch.save(decoder, args.save_dir + 'decoder_%d' % (Epoches))
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch 360 Video Grounding')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--epoches', type=int, default=100, metavar='N',
                        help='number of epoches to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--save_dir', type=str, default='trained_model/',
                        help='folder for storing the model parameters (default: trained_model/)')
    parser.add_argument('--mode', type=str, default='train',
                        help='mode for unsupervised pretraining or supervised finetuning (default: train)')
    parser.add_argument('--video_len', type=str, default='3',
                        help='video clip sample length for training (default: 3)')
    parser.add_argument('--MAX_LENGTH', type=str, default='33',
                        help='subtitle maximum length (default: 33)') ##train 33, val 39, test 30
    args = parser.parse_args()

    ## parameters
    use_cuda = torch.cuda.is_available()
    batch_size = args.batch_size
    MAX_LENGTH = int(args.MAX_LENGTH)
    hidden_size = 300
    decode_size = 256
    video_len = int(args.video_len)
    n_layers = 1
    data_path = 'data/frame_all_fps1/'

    if args.mode == 'train':
        nfov = NFOV(data_path, vocab, video_len, MAX_LENGTH, train=args.mode=='train')
        train_loader = torch.utils.data.DataLoader(nfov, batch_size, shuffle=True, num_workers=8)
        dictionary_size = nfov.get_dictionary_size()

        ''' initialize model '''
        my_encode = model.EncoderRNN(dictionary_size, hidden_size, batch_size, n_layers).cuda()
        my_model = model.NFOV_gorund(dictionary_size, hidden_size, decode_size, MAX_LENGTH, batch_size).cuda()

        ''' load pretrained decoder model '''
        decoder = DecoderRNN(decode_size, 512, len(vocab), 1)
        decoder.load_state_dict(torch.load('models/decoder-5-3000.pkl'))
        if torch.cuda.is_available():
            decoder.cuda()

        ''' mode training '''
        trainEpoches(my_encode, my_model, decoder, MAX_LENGTH, batch_size, args.epoches, video_len, print_every=1)

    elif args.mode == 'test':
        tStart = time.time()
        nfov = NFOV(data_path, vocab, video_len, MAX_LENGTH, train=args.mode=='train')
        test_loader = torch.utils.data.DataLoader(nfov, batch_size, shuffle=True)

        ''' load trained models '''
        my_encode = torch.load(args.save_dir+'/encoder_%d' % args.epoches)
        my_model = torch.load(args.save_dir+'/VGM_%d' % args.epoches)

        ''' testing '''
        recall_final, recall_all, precision_final, precision_all = test(batch_size, args.epoches, my_encode, my_model, MAX_LENGTH)

        tEnd = time.time()
        print("Final recall for %d test videos: %.3f" %(len(os.listdir(data_path)), np.mean(recall_all)))
        print("Final precision for %d test videos: %.3f" %(len(os.listdir(data_path)), np.mean(precision_all)))
        print('Overall time cost: %f ' % (tEnd - tStart))
