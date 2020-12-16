#!/usr/bin/python3

import numpy as np
# import random
import os
from torch.utils import data
from collections import Counter
import torch

RECIPE2ID = {
    "friedegg": 0,
    "coffee": 1,
    "sandwich": 2,
    "cereals": 3,
    "pancake": 4,
    "tea": 5,
    "juice": 6,
    "salat": 7,
    "scrambledegg": 8,
    "milk": 9,
}


def recipe_action_label(vfname, transcript, num_classes):
    action = torch.zeros(num_classes).float()
    for i in range(num_classes):
        if i in transcript:
            action[i] = 1

    recipe = RECIPE2ID[vfname.split("_")[3]]
    recipe = torch.LongTensor([recipe])
    return recipe, action


# self.features[video]: the feature array of the given video (dimension x frames)
# self.transcrip[video]: the transcript (as label indices) for each video
# self.input_dimension: dimension of video features
# self.n_classes: number of classes
class Dataset(object):

    def __init__(self, base_path, video_list, label2index, new_feature=False, add_timestamp=False):
        self.features = dict()
        self.transcript = dict()
        self.gt_label = dict()
        self.add_timestamp = add_timestamp
        self.new_feature = new_feature
        # read features for each video
        base_path = base_path.rstrip('/')
        for video in video_list:
            # video features
            # Becky2:
            if new_feature:  # if new_feature=True i.e train using new set of features
                self.features[video] = np.load(base_path + '/mixed_5c_rgb/' + video + '.npy')
            else:  # old features i.e new_feature=False i.e default case
                self.features[video] = np.load(base_path + '/features/' + video + '.npy')
            # transcript
            with open(base_path + '/transcripts/' + video + '.txt') as f:
                self.transcript[video] = [label2index[line] for line in f.read().split('\n')[0:-1]]
            # gt_label
            with open(base_path + '/groundTruth/' + video + '.txt') as f:
                self.gt_label[video] = [label2index[line] for line in f.read().split('\n')[0:-1]]

        if add_timestamp:
            for video in video_list:
                self.features[video] = self.add_timestamp_func(self.features[video])
            print("Add TimeStamp", self.features[video].shape[0])

        # set input dimension and number of classes
        self.input_dimension = list(self.features.values())[0].shape[0]
        self.n_classes = len(label2index)

        # print("shuffle data: ", self.shuffle)

    def add_timestamp_func(self, sequence):

        N = sequence.shape[1]
        x = [i / (N - 1) for i in range(N)]
        x = np.array(x).reshape([1, N])
        x = 3 * x - 1.5
        sequence = np.concatenate([sequence, x], axis=0)

        return sequence

    def videos(self):
        return self.features.keys()

    def __getitem__(self, video):
        return self.features[video], self.transcript[video], self.gt_label[video]

    def __len__(self):
        return len(self.features)


class DataLoader():

    def __init__(self, dataset, batch_size, shuffle=False):

        self.num_video = len(dataset)
        self.dataset = dataset
        self.videos = list(dataset.videos())
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.num_batch = int(np.ceil(self.num_video / self.batch_size))

        self.selector = list(range(self.num_video))
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.selector)

    def __len__(self):
        return self.num_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.index > self.num_video:
            if self.shuffle:
                np.random.shuffle(self.selector)
            self.index = 0
            raise StopIteration

        else:
            # import ipdb; ipdb.set_trace()
            video_idx = self.selector[self.index: self.index + self.batch_size]
            videos = [self.videos[i] for i in video_idx]
            self.index += self.batch_size

            batch_sequence, batch_action_label, batch_recipe_label = [], [], []
            seq_len = []
            for vfname in videos:
                # print(vfname)
                sequence, trans, _ = self.dataset[vfname]
                sequence = sequence.T
                recipe, action = recipe_action_label(vfname, trans, self.dataset.n_classes)

                seq_len.append(sequence.shape[0])
                batch_sequence.append(sequence)
                batch_action_label.append(action)
                batch_recipe_label.append(recipe)

            max_seq_len = max(seq_len)
            tensor_seq = np.zeros([len(videos), max_seq_len, batch_sequence[0].shape[1]], dtype=np.float32)
            for i, seq in enumerate(batch_sequence):
                tensor_seq[i, :seq_len[i]] = seq

            seq_len = torch.LongTensor(seq_len)
            # batch_sequence = np.stack(batch_sequence, axis=0)
            tensor_seq = torch.from_numpy(tensor_seq)
            tensor_action_label = torch.stack(batch_action_label, dim=0)
            tensor_recipe_label = torch.stack(batch_recipe_label, dim=0)[:, 0]

            return tensor_seq, seq_len, tensor_action_label, tensor_recipe_label





