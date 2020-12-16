#!/usr/bin/python3

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


# save best A_noBG_F1 checkpoint. func takes in the dict checkpoint and saves it to filepath
def save_best_checkpoint(state,
                         filepath='/home/becky/checkpoints/baseline_on_new_features/best_A_noBG_F1_checkpoint.pt'):
    print(" => saving best A_noBG_F1 checkpoint")
    print(" ")
    torch.save(state, filepath)


def compute_f1(label, pred):
    correct_pos = (label * pred).sum(0)
    pred_pos = pred.sum(0)
    true_pos = label.sum(0)

    precision = correct_pos / (pred_pos + 1e-5)
    recall = correct_pos / (true_pos + 1e-5)

    f1 = (2 * precision * recall) / (precision + recall + 1e-5)
    return f1


test_R_losses = []
test_A_losses = []

test_R_accuracy = []
test_A_noBG_accuracy = []
test_A_noBG_F1 = []


def evaluate(net, testloader, recorder, pos_weight):
    print("TESTING" + "~" * 10)
    with torch.no_grad():
        for batch_idx, (batch_seq, seq_len, action_label, recipe_label) in enumerate(testloader):
            batch_seq = batch_seq.cuda()
            seq_len = seq_len.cuda()
            action_label = action_label.cuda()
            recipe_label = recipe_label.cuda()

            net.forward_and_loss(batch_seq, seq_len, action_label, recipe_label, pos_action_weight=pos_weight)

            recorder.append('test_recipe_loss', net.recipe_loss.item() * net.recipe_weight)
            recorder.append('test_action_loss', net.action_loss.item())

            recorder.append('test_action_noBG_acc', net.action_acc)
            recorder.append('test_action_pred', net.action_pred.detach().cpu().numpy())
            recorder.append('test_action_label', action_label.detach().cpu().numpy())
            recorder.append('test_recipe_acc', net.recipe_acc)

    action_pred = np.concatenate(recorder.get_reset("test_action_pred"), axis=0)
    action_label = np.concatenate(recorder.get_reset("test_action_label"), axis=0)
    f1 = compute_f1(action_label, action_pred)
    f1_noBG = f1[1:].mean()

    t_r_loss = recorder.mean_reset('test_recipe_loss')
    test_R_losses.append(t_r_loss)
    string = "R_loss: %.3f, " % t_r_loss
    t_a_loss = recorder.mean_reset('test_action_loss')
    test_A_losses.append(t_a_loss)
    string += "A_loss: %.3f, " % t_a_loss
    print(string)
    t_r_acc = recorder.mean_reset('test_recipe_acc')
    test_R_accuracy.append(t_r_acc)
    string = "R_acc: %.3f, " % t_r_acc
    t_anoBG_acc = recorder.mean_reset('test_action_noBG_acc')
    test_A_noBG_accuracy.append(t_anoBG_acc)
    string += "A_noBG_acc: %.3f, " % t_anoBG_acc
    test_A_noBG_F1.append(f1_noBG)
    string += "A_noBG_F1: %.3f, " % f1_noBG
    print(string + "\n")


parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--recipe_weight', default=1.0, type=float, help='')
parser.add_argument('--resume', default=None, type=str, help='path of model to resume')
parser.add_argument('--epoch', default=10000, type=int, help='number of total epochs to run')
parser.add_argument('--data', default="breakfast", type=str)
# parser.add_argument('--batch_size', default=32, type=int)
# divide batch_size/2 to help with the CUDA memory issues
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

# print('output of gru layer vs input of linear layer, see the size of that matrix, it should match:' + )

### read training data #########################################################
print('read data...')
with open(train_split_fname, 'r') as f:
    video_list = f.read().split('\n')[0:-1]
dataset = Dataset(dataset_dir, video_list, label2index, new_feature=args.new_feature)
trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

with open(test_split_fname, 'r') as f:
    test_video_list = f.read().split('\n')[0:-1]
test_dataset = Dataset(dataset_dir, test_video_list, label2index, new_feature=args.new_feature)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("Number of training data", len(video_list))
print(dataset)

# negative label dominates
# so give larger weight to positive label
action_occur_frequency = np.loadtxt(os.path.join(dataset_dir, "action_occurence.txt"))
missing_frequency = 1 - action_occur_frequency
pos_weight = missing_frequency / action_occur_frequency
pos_weight[0] = 1  # set weight of Background to normal level
pos_weight = torch.FloatTensor(pos_weight).cuda()

### create network #########################################################
net = Net(dataset.input_dimension, args.hidden_size, n_class, len(RECIPE2ID),
          recipe_weight=args.recipe_weight, no_gru=args.no_gru)
net.cuda()
print(net)

start_epoch = 0
if args.resume:
    assert ("Split%d" % args.split in args.resume)
    load_iteration = os.path.basename(args.resume)
    load_iteration = int(load_iteration.split('.')[1].split('-')[1])
    args.resume = os.path.dirname(args.resume)
    print("Load from %s, Iteraion %d" % (args.resume, load_iteration))

    state_dict = torch.load(args.resume + '/network.iter-' + str(load_iteration) + '.net')
    neq_load_customized(net, state_dict)
    print("Resume Iteration!")
    start_epoch = load_iteration

    if load_iteration > args.lr_decay_iter:
        args.lr = args.lr * 0.1

recorder = Recorder()

# best weight_decay
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# instantiate optimizer
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

train_R_losses = []
train_A_losses = []

train_R_accuracy = []
train_A_noBG_accuracy = []
train_A_noBG_F1 = []

epochs = range(start_epoch, args.epoch)
train_epochs = []
test_epochs = []

global_step = 0
for eidx in range(start_epoch, args.epoch):

    # save checkpoint as state_dict i.e model parameters, optimizer and best A_noBG_F1
    # if eidx == 60:
    #     checkpoint = {'state_dict' : net.state_dict(), 'optimizer' : optimizer.state_dict() } #, 'f1_noBG' : f1_noBG}
    #     save_best_checkpoint(checkpoint)

    for batch_idx, (batch_seq, seq_len, action_label, recipe_label) in enumerate(trainloader):
        batch_seq = batch_seq.cuda()
        seq_len = seq_len.cuda()
        action_label = action_label.cuda()
        recipe_label = recipe_label.cuda()

        optimizer.zero_grad()
        net.forward_and_loss(batch_seq, seq_len, action_label, recipe_label, pos_action_weight=pos_weight)
        net.loss.backward()
        optimizer.step()

        # tracking training statistic
        recorder.append('recipe_loss', net.recipe_loss.item() * net.recipe_weight)
        recorder.append('action_loss', net.action_loss.item())

        recorder.append('action_noBG_acc', net.action_acc)
        recorder.append('action_pred', net.action_pred.detach().cpu().numpy())
        recorder.append('action_label', action_label.detach().cpu().numpy())
        recorder.append('recipe_acc', net.recipe_acc)

        # print some progress information
        if (global_step + 1) % 20 == 0:
            action_pred = np.concatenate(recorder.get_reset("action_pred"), axis=0)
            action_label = np.concatenate(recorder.get_reset("action_label"), axis=0)
            f1 = compute_f1(action_label, action_pred)
            f1_noBG = f1[1:].mean()

            string = "Iteration %d, " % (global_step + 1)
            r_loss = recorder.mean_reset('recipe_loss')
            train_R_losses.append(r_loss)
            string += "R_loss: %.3f, " % r_loss
            a_loss = recorder.mean_reset('action_loss')
            train_A_losses.append(a_loss)
            string += "A_loss: %.3f, " % a_loss
            print(string)
            string = " " * len("Iteration %d, " % (global_step + 1))
            r_acc = recorder.mean_reset('recipe_acc')
            train_R_accuracy.append(r_acc)
            string += "R_acc: %.3f, " % r_acc
            a_noBG_acc = recorder.mean_reset('action_noBG_acc')
            train_A_noBG_accuracy.append(a_noBG_acc)
            string += "A_noBG_acc: %.3f, " % a_noBG_acc
            train_A_noBG_F1.append(f1_noBG)
            string += "A_noBG_F1: %.3f, " % f1_noBG
            print(string + "\n")

            train_epochs.append(global_step + 1)  # saving the number of iterations

        # test and save model every 500 iterations
        if (global_step + 1) % 500 == 0:
            net.eval()  # turn off dropout at test time
            evaluate(net, testloader, recorder, pos_weight)
            net.train()  # turn on dropout at train time
            print('save snapshot ' + str(eidx + 1), args.exp)
            network_file = savedir + '/network.iter-' + str(eidx + 1) + '.net'
            net.save_model(network_file)

            test_epochs.append(global_step + 1)  # saving the number of iterations

        global_step += 1

        # if global_step == 6040:
        #     break

    # adjust learning rate after 2500 iterations
    if args.lr_decay_iter != 0 and (eidx + 1) == args.lr_decay_iter:
        args.lr = args.lr * 0.1


def draw_loss_curve(train_epochs, train_R_losses, train_A_losses, test_epochs, test_R_losses, test_A_losses, title):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_epochs, train_R_losses, color='green', label='Train R_loss')
    plt.plot(train_epochs, train_A_losses, color='blue', label='Train A_loss')
    plt.plot(test_epochs, test_R_losses, color='orange', marker='o', label='Test R_loss')
    plt.plot(test_epochs, test_A_losses, color='yellow', marker='o', label='Test A_loss')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title(title, fontsize=24)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Losses', fontsize=16)
    plt.legend(loc='upper right', fontsize=14)
    plt.savefig('/home/becky/plots/new_features/baseline_' + title + '.png')
    # plt.show()


def draw_accuracy_curve(train_epochs, train_R_accuracy, train_A_noBG_accuracy, train_A_noBG_F1, test_epochs,
                        test_R_accuracy, test_A_noBG_accuracy, test_A_noBG_F1, title):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(train_epochs, train_R_accuracy, color='green', label='Train R_accuracy')
    plt.plot(train_epochs, train_A_noBG_accuracy, color='blue', label='Train A_noBG_accuracy')
    plt.plot(train_epochs, train_A_noBG_F1, color='purple', label='Train A_noBG_F1')
    plt.plot(test_epochs, test_R_accuracy, color='orange', marker='o', label='Test R_accuracy')
    plt.plot(test_epochs, test_A_noBG_accuracy, color='yellow', marker='o', label='Test A_noBG_accuracy')
    plt.plot(test_epochs, test_A_noBG_F1, color='red', marker='o', label='Test A_noBG_F1')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.title(title, fontsize=24)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left', fontsize=14)
    plt.savefig('/home/becky/plots/new_features/baseline_' + title + '.png')
    # plt.show()


# import ipdb; ipdb.set_trace()
# draw train/test loss curve as the number of epochs increases -for both loss terms
draw_loss_curve(train_epochs, train_R_losses, train_A_losses, test_epochs, test_R_losses, test_A_losses,
                'Train_and_Test_Losses')
# draw train/test accuracy curve as the number of epochs increases -for all accuracy terms
draw_accuracy_curve(train_epochs, train_R_accuracy, train_A_noBG_accuracy, train_A_noBG_F1, test_epochs,
                    test_R_accuracy, test_A_noBG_accuracy, test_A_noBG_F1, 'Train_and_Test_Accuracies')

print('FINISHED')