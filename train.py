import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sklearn
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import argparse
np.set_printoptions(suppress=True)
# tensorboard
from tensorboardX import SummaryWriter

from src.loss import customized_loss, margin_ranking_loss
from src.dataset import Dataset
from src.layers import GraphConvLayer
from src.utils import generate_neg_sample, load_data
from src.model import Model

import logging
from datetime import datetime
import time

EXP_NAME = f'exp_{time.time_ns()}'

def logger_init(log_file_name='monitor',
                log_level=logging.DEBUG,
                log_dir='./logs/',
                only_file=False):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '_' + EXP_NAME + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    if only_file:
        logging.basicConfig(filename=log_path,
                            level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,
                            format=formatter,
                            datefmt='%Y-%d-%m %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),
                                      logging.StreamHandler(sys.stdout)]
                            )

logger_init()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"current device is {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=500, help="epoch to run")
parser.add_argument("--seed", type=int, default=8, help="training set ratio")
parser.add_argument('--hidden', type=int, default=128, help="hidden dimension of entity embeddings")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--k', type=float, default=10, help="hit@k")
parser.add_argument('--negsize', type=int, default=10, help="number of negative samples")
parser.add_argument('--negiter', type=int, default=10, help="re-calculate epoch of negative samples")
parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay coefficient")
parser.add_argument('--graph_s', type=str, default="data_G1", help="source graph path")
parser.add_argument('--graph_d', type=str, default="data_G2", help="destination graph path")
parser.add_argument('--anoise', type=float, default=0.2, help="anchor noise")
parser.add_argument('--board_path', type=str, default='board', help="tensorboard path")
args = parser.parse_args()

############################
# parameters
epoch = args.epoch
embedding_dim = args.hidden
learning_rate = args.lr
weight_decay = args.weight_decay
neg_samples_size = args.negsize
negiter = args.negiter
graph_path_s = args.graph_s
graph_path_d = args.graph_d
train_seeds_ratio = args.seed * 0.1
k = args.k
anoise = args.anoise
############################
tb_logger = SummaryWriter(args.board_path)

############################
# preprocess
graph1 = graph_path_s
graph2 = graph_path_d
A1, A2, anchor = load_data(graph1=graph1, graph2=graph2, anoise=anoise)
train_size = int(train_seeds_ratio * len(anchor[:, 0]))
test_size = len(anchor[:, 0]) - train_size
train_set, test_set = torch.utils.data.random_split(anchor, lengths=[train_size, test_size])
train_set = np.array(list(train_set))
test_set = np.array(list(test_set))
batchsize = train_size
train_dataset = Dataset(train_set)
train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_adjust_freq, args.lr_decay_rate)
criterion = nn.TripletMarginLoss(margin=3, p=2)
############################

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info(f'#total params: {pytorch_total_params}')
logging.info(f"training samples: {train_size}, test samples: {test_size}")
logging.info(f"model architecture:\n {model}")


def predict(output_file, sim_measure="cosine",):
    """
    将预测写入文件
    """
    model.eval()
    Embedding1, Embedding2 = model()
    Embedding1 = Embedding1.detach()
    Embedding2 = Embedding2.detach()

    # step 1: generate sim mat
    if sim_measure == "cosine":
        # similarity_matrix = cosine_similarity(Embedding1, Embedding2)
        similarity_matrix = torch.mm(Embedding1, Embedding2.t()).cpu().numpy()
    else:
        Embedding1 = Embedding1.numpy()
        Embedding2 = Embedding2.numpy()
        similarity_matrix = euclidean_distances(Embedding1, Embedding2)
        similarity_matrix = np.exp(-similarity_matrix)
    # step 2: information statistics
    # alignment_hit1 = list()
    file = open(output_file, 'w')
    for idx0, line in enumerate(similarity_matrix):
        idx = np.argmax(line)
        idx = int(idx)
        # alignment_hit1.append(idx)
        file.write(f'{idx0} {idx}\n')


def evaluate(data, k, sim_measure="cosine", phase="test"):
    model.eval()
    Embedding1, Embedding2 = model()
    Embedding1 = Embedding1.detach()
    Embedding2 = Embedding2.detach()
    if phase == "over":
        logging.info(Embedding1)
    # step 1: generate sim mat
    if sim_measure == "cosine":
        # similarity_matrix = cosine_similarity(Embedding1, Embedding2)
        similarity_matrix = torch.mm(Embedding1, Embedding2.t()).cpu().numpy()
    else:
        Embedding1 = Embedding1.numpy()
        Embedding2 = Embedding2.numpy()
        similarity_matrix = euclidean_distances(Embedding1, Embedding2)
        similarity_matrix = np.exp(-similarity_matrix)
    # step 2: information statistics
    alignment_hit1 = list()
    alignment_hitk = list()
    for line in similarity_matrix:
        idx = np.argmax(line)
        idx = int(idx)
        alignment_hit1.append(idx)
        idxs = heapq.nlargest(k, range(len(line)), line.take)
        alignment_hitk.append(idxs)
    # step 3: calculate evaluate score: hit@1 and hit@k
    hit_1_score = 0
    hit_k_score = 0
    for idx in range(len(data)):
        g1_idx = int(data[idx][0])
        gt = data[idx][1]  # todo
        if int(gt) == alignment_hit1[g1_idx]:
            hit_1_score += 1
        if int(gt) in alignment_hitk[g1_idx]:
            hit_k_score += 1
    return similarity_matrix, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score


# begin training
best_E1 = None
best_E2 = None
best_hit_1_score = 0
neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(train_set, neg_samples_size=neg_samples_size)
for e in range(epoch):
    model.train()
    if e % negiter == 0:
        neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(train_set, neg_samples_size=neg_samples_size)
    for _, data in enumerate(train_loader):
        a1_align, a2_align = data
        E1, E2 = model()
        optimizer.zero_grad()
        loss = customized_loss(E1, E2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right, neg_samples_size=neg_samples_size, neg_param=0.3)
        # loss = margin_ranking_loss(criterion, E1, E2, a1_align, a2_align, neg1_left, neg1_right, neg2_left, neg2_right)
        loss.backward()  # print([x.grad for x in optimizer.param_groups[0]['params']])
        optimizer.step()
        scheduler.step()
        sim_mat, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score = evaluate(data=test_set, k=k)

        if hit_1_score > best_hit_1_score:
            best_hit_1_score = hit_1_score
            # todo save model
            logging.info(
                f"current best Hits@1 count {hit_1_score}({round(hit_1_score/len(test_set), 2)}) ,hit@{k}: total {hit_k_score}({round(hit_k_score/len(test_set), 2)}) at the {e+1}th epoch: {best_hit_1_score}")

    tb_logger.add_scalar('loss_train', loss.item(), e + 1)
    logging.info(f"epoch: {e+1}, loss: {round(loss.item(), 3)}\n")

# final evaluation and test
# ground_truth = np.loadtxt('ground_truth.txt', delimiter=' ')
# ground_truth = np.loadtxt('data/anchor/anchor_0.2_test.txt', delimiter=' ')
# similarity_matrix, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score = evaluate(data=ground_truth, k=k, phase="over")
# logging.info(similarity_matrix)
# logging.info(f"final score: hit@1: total {hit_1_score} and ratio {round(hit_1_score/len(ground_truth), 2)}, hit@{k}: total {hit_k_score} and ratio {round(hit_k_score/len(ground_truth), 2)}")

# 写入文件
predict(f'submit_tmp_{args.graph_s}_{args.graph_d}_{anoise}.txt',)
