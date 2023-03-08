import numpy as np
import os
import sys
import pickle
import torch
import random
import slowfast.utils.distributed as du
import slowfast.utils.checkpoint as cu
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter
import cv2
import pandas as pd
import tqdm
"""Wrapper to train and test a video classification model."""
from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
import time
import csv
from itertools import islice

class VideoReader(object):
    def __init__(self, source):
        self.source = source
        try:  # OpenCV needs int to read from webcam
            self.source = int(source)
        except ValueError:
            pass
    def __iter__(self):
        self.cap = cv2.VideoCapture(self.source)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.source))
        return self
    def __next__(self):
        was_read, frame = self.cap.read()
        if not was_read:
            # raise StopIteration
            ## reiterate the video instead of quiting.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = None
            # print('end video')
        return was_read, frame
    def clean(self):
        self.cap.release()
        cv2.destroyAllWindows()

def main(cfg, videoids, labels, path, checkpoint_list):
    print(videoids)
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[0]
    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)
    model.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[1]
    model_2 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_2)
    model_2.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[2]
    model_3 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_3)
    model_3.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[3]
    model_4 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_4)
    model_4.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    cfg.TEST.CHECKPOINT_FILE_PATH = checkpoint_list[4]
    model_5 = build_model(cfg)
    cu.load_test_checkpoint(cfg, model_5)
    model_5.eval()
    print('cfg', cfg.TEST.CHECKPOINT_FILE_PATH)

    total_prob = {}
    video_order = []

    for key, values in videoids.items():
        video_order.append(values)
        video_path = values[1]
        print(video_path)
        img_provider = VideoReader(video_path)
        fps = 30
        print('fps:', fps)
        frames = []
        s = 0.
        count = 0
        print(cfg.DATA.NUM_FRAMES, cfg.DATA.SAMPLING_RATE)
        predict_sq = []
        prob_sq = []
        score_sq = []
        for able_to_read, frame in img_provider:
            count += 1
            if not able_to_read:
                frames = []
                break
            if len(frames) != cfg.DATA.NUM_FRAMES and count % cfg.DATA.SAMPLING_RATE ==0:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed = cv2.resize(frame_processed, (512, 512), interpolation = cv2.INTER_AREA)
                frames.append(frame_processed)
            if len(frames) == cfg.DATA.NUM_FRAMES:
                start = time.time()
                inputs = torch.tensor(np.array(frames)).float()
                inputs = inputs / 255.0
                inputs = inputs - torch.tensor(cfg.DATA.MEAN)
                inputs = inputs / torch.tensor(cfg.DATA.STD)
                inputs = inputs.permute(3, 0, 1, 2)
                inputs = inputs[None, :, :, :, :]
                index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
                fast_pathway = torch.index_select(inputs, 2, index)
                inputs = [inputs]
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                preds  = model(inputs).detach().cpu().numpy()   
                preds_2  = model_2(inputs).detach().cpu().numpy()   
                preds_3  = model_3(inputs).detach().cpu().numpy()   
                preds_4  = model_4(inputs).detach().cpu().numpy()   
                preds_5  = model_5(inputs).detach().cpu().numpy()   
                prob_ensemble = np.array([preds, preds_2, preds_3, preds_4, preds_5])
                prob_ensemble = np.mean(prob_ensemble, axis=0)
                prob_sq.append(prob_ensemble)
                frames = []
        print(prob_sq)
        total_prob[values[0]] = prob_sq
    return dict(sorted(total_prob.items())), video_order

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_classification(sequence_class_prob):
    classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1) 
    probs= np.max(sequence_class_prob, axis=1)
    return labels_index,

if __name__ == "__main__":
    args = parse_args()
    path_to_config = args.cfg_files[0]
    cfg = load_config(args, path_to_config)
    cfg = assert_and_infer_cfg(cfg)
    fps = 30
    seed(42)
    labels = list(range(0, 16))
    # print(labels)
    path = sys.argv[8]
    # print(path)
    video_ids={}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[1]] = row[0]
                video_names.append(row[1])
    # print(video_names)
    import glob
    text_files = glob.glob(path+"/**/*.MP4", recursive=True)
    # print(text_files)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    # print(vid_info)
    checkpoint_dashboard_list=[
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Dashboard_group_0/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Dashboard_group_1/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Dashboard_group_2/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Dashboard_group_3/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Dashboard_group_4/checkpoints/checkpoint_epoch_00010.pyth',
    ]
    vid_info = dict(sorted(vid_info.items()))
    prob_1, video_order = main(cfg, vid_info, labels, filelist, checkpoint_dashboard_list)
    print(prob_1)
    print(video_order)

    video_ids={}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[2]] = row[0]
                video_names.append(row[2])
    text_files = glob.glob(path+"/**/*.MP4", recursive=True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    
    checkpoint_dashboard_list=[
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Rear_view_group_0/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Rear_view_group_1/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Rear_view_group_2/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Rear_view_group_3/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Rear_view_group_4/checkpoints/checkpoint_epoch_00010.pyth',
    ]
    vid_info = dict(sorted(vid_info.items()))
    prob_2, video_order = main(cfg, vid_info, labels, filelist, checkpoint_dashboard_list)

    video_ids={}
    video_names = []
    with open(os.path.join(path, 'video_ids.csv')) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(csvReader):
            if idx > 0:
                video_ids[row[3]] = row[0]
                video_names.append(row[2])
    text_files = glob.glob(path+"/**/*.MP4", recursive=True)
    filelist = {}
    for root, dirs, files in os.walk(path):
        for vid_name in files:
            if vid_name in video_names:
                filelist[vid_name] = os.path.join(root, vid_name)
    vid_info = {}
    for key in (video_ids.keys() | filelist.keys()):
        if key in video_ids: vid_info.setdefault(key, []).append(video_ids[key])
        if key in filelist: vid_info.setdefault(key, []).append(filelist[key])
    
    checkpoint_dashboard_list=[
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Right_side_window_group_0/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Right_side_window_group_1/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Right_side_window_group_2/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Right_side_window_group_3/checkpoints/checkpoint_epoch_00010.pyth',
        '/content/drive/MyDrive/AICC2023-Track3/checkpoint_Right_side_window_group_4/checkpoints/checkpoint_epoch_00010.pyth',
    ]
    vid_info = dict(sorted(vid_info.items()))
    prob_2, video_order = main(cfg, vid_info, labels, filelist, checkpoint_dashboard_list)