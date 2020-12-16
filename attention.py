import numpy as np
from dataset import Dataset, recipe_action_label, RECIPE2ID, DataLoader
from network import Net
from .utils.utils import get_dataset_paths, load_action_mapping, neq_load_customized, Recorder
from .home import get_project_base
from .utils.general import prepare_save_env
import argparse
import os
from torch.utils import data
from torch import optim
import torch
import pickle
import matplotlib.pyplot as plt


# load best A_noBG_F1 checkpoint
def load_best_checkpoint(checkpoint_path, net, optimizer):
    # checkpoint_path: path to save checkpoint
    # net: model that we want to load checkpoint parameters into
    # optimizer: optimizer we defined in previous training

    print(" ")
    print(" => loading best A_noBG_F1 checkpoint: " + checkpoint_path)
    print(" ")
    # load check point
    checkpoint = torch.load(checkpoint_path)
    # initialize state_dict from checkpoint to model
    net.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    net.eval()  # turn off dropout
    return net, optimizer


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--recipe_weight', default=1.0, type=float, help='')
parser.add_argument('--resume', default=None, type=str, help='path of model to resume')
parser.add_argument('--epoch', default=10000, type=int, help='number of total epochs to run')
parser.add_argument('--data', default="breakfast", type=str)
parser.add_argument('--batch_size', default=16, type=int)  # 16
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lr_decay_iter', default=2500, type=int, help='learning rate')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--exp', default='tmp', type=str, help='prefix of checkpoint filename')
parser.add_argument('--split', default=1, type=int)
parser.add_argument('--no_gru', action="store_true")
parser.add_argument('--new_feature', action="store_true")

args = parser.parse_args()
BASE = get_project_base()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

args.exp = "Split%d/recipeWeight%.3f_%s" % (args.split, args.recipe_weight, args.exp)
base_log_dir = "log/Breakfast/"
logger, logdir, savedir, _ = prepare_save_env(BASE + base_log_dir, args.exp, args,
                                              tensorboard=False,
                                              stdout=True)  # stdout True -> save all print to log file

map_fname, dataset_dir, train_split_fname, test_split_fname = get_dataset_paths(args.data, args.split)
label2index, index2label = load_action_mapping(map_fname)
n_class = len(label2index)
print("load_data_from", dataset_dir)
print(" ")

# read testing videos
print('read test data...')
with open(test_split_fname, 'r') as f:
    test_video_list = f.read().split('\n')[0:-1]
test_dataset = Dataset(dataset_dir, test_video_list, label2index, new_feature=args.new_feature)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("Number of testing data", len(test_video_list))
print(test_dataset)

# negative label dominates
# so give larger weight to positive label
action_occur_frequency = np.loadtxt(os.path.join(dataset_dir, "action_occurence.txt"))
missing_frequency = 1 - action_occur_frequency
pos_weight = missing_frequency / action_occur_frequency
pos_weight[0] = 1  # set weight of Background to normal level
pos_weight = torch.FloatTensor(pos_weight).cuda()

# create network
net = Net(test_dataset.input_dimension, args.hidden_size, n_class, len(RECIPE2ID),
          recipe_weight=args.recipe_weight, no_gru=args.no_gru)
net.cuda()

# define optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# define checkpoint saved path
checkpoint_path = "./checkpoints/best_dropout_best_weight_decay_on_new_features/best_A_noBG_F1_checkpoint.pt"

# load the saved best A_noBG_F1 checkpoint i.e load this model for inference
net, optimizer = load_best_checkpoint(checkpoint_path, net, optimizer)

print("net = ", net)
print(" ")
print("optimizer = ", optimizer)
print(" ")

# save attn output to visualize
temporal_attention_all_video = []

# inference : run loaded model on test videos
# samples => batch_seq, seq_len, and labels => action_label, recipe_label
for batch_seq, seq_len, action_label, recipe_label in testloader:
    with torch.no_grad():
        batch_seq = batch_seq.cuda()
        seq_len = seq_len.cuda()
        action_label = action_label.cuda()
        recipe_label = recipe_label.cuda()

    # temporal_attention gives me the attn for all actions and all frames for all videos i.e in this batch
    action_logit, recipe_logit, temporal_attention = net(batch_seq, seq_len)  # i.e output = net(samples)

    # saving intermediate results i.e the temporal_attention_output for all test videos/'batches'
    temporal_attention_all_video.append(temporal_attention)

# temporal_attention is of the shape/dimension batch_size x sequence/number of frames x number of actions/classes BTC
B, T, C = temporal_attention.shape

count_videos = 0

# loop over all batch in list. batch_i is the ith batch of test videos
for batch in temporal_attention_all_video:
    X, Y, Z = batch.shape

    # process each video in this batch
    for i in range(0, X):
        video_i_attention = batch[i]
        count_videos = count_videos + 1

        # transcript and label of video i.e the transcript (as label indices) for each video
        _, transcript, framelabel = test_dataset[testloader.videos[i]]

        action_idx = np.unique(transcript)
        # get the indices of actions happened in the video
        # Eg. if only action 0, 4, 8 happened in the video, action_idx = [0, 4, 8]

        # transcript = [0, 1, 2, 0 ]
        # action_idx = [0, 1, 2 ]

        # transform label = [0,0,0,1,1,1] to one-hot form i.e pass the label into its one-hot format
        onehot_label = np.zeros_like(
            video_i_attention.cpu().detach())  # can't call numpy on tensor that requires grad
        # onehot_label = np.zeros_like(video_i_attention  # throws cuda error so transfer data to cpu before converting to numpy

        E, F = onehot_label.shape
        if len(framelabel) <= E:  # ignore videos that throw an IndexError
            onehot_label[list(range(len(framelabel))), framelabel] = 1

            # compare l_i with p(a_i) and compute the quantitative score to measure quality of attention
            score_summation = 0
            score = 0

            fig, axs = plt.subplots(2, 1)
            for a in action_idx:
                # use temporal_attention[0, :, 0] to find the corresponding vector and plot it

                axs[0].plot(video_i_attention[:, a].cpu().detach(), label=index2label[a], c=plt.cm.Paired(a))
                axs[1].plot(onehot_label[:, a], label=index2label[a] + "_label", c=plt.cm.Paired(a))

                score_dot_product = torch.matmul(torch.from_numpy(onehot_label[:, a]).cuda(),
                                                 video_i_attention[:, a])  # dot product of l_i and p(a_i)
                # score_dot_product = score_dot_product/E  # normalize by number of frames
                score_summation = score_summation + score_dot_product  # sum up score for all actions
            score = score_summation / F  # normalize by number of actions. score is 0 dim here
            # print("score for " + testloader.videos[i] + " is " + str(score.item()))
            title = str(score.item()) + '_' + testloader.videos[i]

            axs[0].set_xlim(0, len(framelabel))
            axs[0].set_ylabel('Attention', fontsize=10)
            axs[0].legend(loc='upper right', fontsize=6)

            axs[1].set_xlim(0, len(framelabel))
            axs[1].set_xlabel('Frame', fontsize=10)
            axs[1].set_ylabel('Label', fontsize=10)
            axs[1].legend(loc='upper right', fontsize=6)

            plt.savefig('/home/becky/plots/new_features_best_dropout_best_weight_decay/' + title + '.jpg')
            plt.cla()
            # python -m src.attention --new_feature --gpu 1

    score = score / count_videos  # average over all videos
    print("the overall attention score is: " + str(score.item()))

