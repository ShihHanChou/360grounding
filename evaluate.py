import numpy as np
import subprocess
import os
from collections import namedtuple
import VideoRecorder_evaluate as VideoRecorder
from PIL import Image

def get_label_from_txt(label_dir, img_root, sample_size):
    label_all = []
    object_name = []
    for ele in np.sort(os.listdir(label_dir)):
        if len(ele.split('.')) > 1 and ele.split('.')[1] == 'xml':
            tmp = open(label_dir+'/'+ele).read()
            vid = ele.split('.')[0]
            frame_all = len(os.listdir(img_root+vid))*sample_size
                            
            seg_object = tmp.split('<attribute name="')[5:]
            label =  np.zeros([len(seg_object),frame_all,4])

            for seg_object_id in range(len(seg_object)):
                object_name.append(seg_object[seg_object_id].split('"')[0])
                seg_frame = seg_object[seg_object_id].split('<data:polygone')[1:]
                label_axis = np.zeros([frame_all,4])
                for seg_frame_id in range(len(seg_frame)):
                    frame_num = int(seg_frame[seg_frame_id].split('framespan="')[1].split(':')[0])
                    x_axis = int(seg_frame[seg_frame_id].split('x="')[1].split('"')[0])
                    y_axis = int(seg_frame[seg_frame_id].split('y="')[1].split('"')[0])
                    width = int(seg_frame[seg_frame_id].split('width="')[1].split('"')[0])
                    height = int(seg_frame[seg_frame_id].split('height="')[1].split('"')[0])
                    label_axis[frame_num-1,:] = [x_axis, y_axis, x_axis+width, y_axis+height]
                label[seg_object_id,:]=label_axis
        label_all.append([vid,label])
    return label_all


def evaluate(guess_o, video_idx, img_size, IOU_thre, sample_size = 30):

    img_data = 'data/frame_all_fps1/'
    label_all = get_label_from_txt('data/label_test', img_data, sample_size)
    #Xmin, Ymin, Xmax, Ymax of guess NFOV
    guess = np.zeros((guess_o[:,:,0].shape[0],guess_o[:,:,0].shape[1],4))
    sphereH = img_size[0] ## 720
    sphereW = img_size[1] ## 1280
    warp_img = VideoRecorder.ImageRecorder(sphereW, sphereH)
    
    Avg_IOU = []
    Avg_recall_batch = []
    Avg_precision_batch = []
    all_idx = []
    Avg_IOU_each_file = []
    Avg_IOU_each_file_pre = []
    for batch_idx in video_idx:
        dir_name = label_all[batch_idx][0]
        IOU_each_frame = []
        IOU_each_frame_pre = []

        for jj in range(len(os.listdir(img_data+dir_name))*sample_size): # frame
            if jj % sample_size == 0:
                img = np.zeros(img_size)
                bth_idx = np.where(video_idx==batch_idx)[0][0]
                frame_idx = jj/sample_size
                Px, Py = warp_img._sample_points(guess_o[bth_idx, frame_idx, 0], guess_o[bth_idx, frame_idx, 1])
                Px[np.where(np.greater_equal(Px,sphereW-1))] = 0
                Py[np.where(np.greater_equal(Py,sphereH-1))] = 0
                img[Py.astype(int), Px.astype(int)] = 1
                img[Py.astype(int)+1, Px.astype(int)+1] = 1
                sum_NFOV = np.sum(img)

                img_o = np.zeros(img_size)
                for kk in range(len(label_all[batch_idx][1])): # object
                    if np.all(label_all[batch_idx][1][kk][jj] != 0.0):
                        ymin, ymax = int(label_all[batch_idx][1][kk][jj][1]), int(label_all[batch_idx][1][kk][jj][3])
                        xmin, xmax = int(label_all[batch_idx][1][kk][jj][0]), int(label_all[batch_idx][1][kk][jj][2])
                        img_o[ymin:ymax, xmin:xmax]=1
                        img[ymin:ymax, xmin:xmax]=1
                sum_GT = np.sum(img_o)
                sum_union = np.sum(img)
                
                if sum_GT > 0.0:
                    IOU_each_frame.append((sum_NFOV + sum_GT - sum_union)/sum_GT)
                    IOU_each_frame_pre.append((sum_NFOV + sum_GT - sum_union)/sum_NFOV)

        if IOU_each_frame != []:
            Avg_IOU_each_file.append(sum(IOU_each_frame)/len(IOU_each_frame))
        if IOU_each_frame_pre != []:
            Avg_IOU_each_file_pre.append(sum(IOU_each_frame_pre)/len(IOU_each_frame_pre))

        all_idx.append(batch_idx)
    Avg_recall_batch.append(np.mean(Avg_IOU_each_file))
    Avg_precision_batch.append(np.mean(Avg_IOU_each_file_pre))
    return Avg_recall_batch, all_idx, Avg_precision_batch
