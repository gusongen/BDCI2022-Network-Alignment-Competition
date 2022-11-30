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
import random
import sys
from sklearn.model_selection import KFold

EXP_NAME = f'exp_{time.time_ns()}'


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # dgl.random.seed(seed)


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
seed(2022)
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
parser.add_argument('--lr_adjust_freq', default=100, type=int, help='decay lr after certain number of epoch')
parser.add_argument('--lr_decay_rate', default=0.8, type=float, help='learning rate decay')
parser.add_argument('--cv', default=5, type=int, help='k-fold cross validation')
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
# os.makedirs(args.board_path + '/' + EXP_NAME, exist_ok=True)
# tb_logger = SummaryWriter(args.board_path + '/' + EXP_NAME)

############################
# preprocess
graph1 = graph_path_s
graph2 = graph_path_d
A1, A2, anchor = load_data(graph1=graph1, graph2=graph2, anoise=anoise)

def generate_train_set_with_threashold(train_set,a1_embedding,a2_embedding,rate=1):
    similarity_martix=a1_embedding.detach().numpy().dot(a2_embedding.detach().numpy().T)
    thresholded_align=(-similarity_martix[train_set[:,0].astype('int'),train_set[:,1].astype('int')]).argsort()[:int(len(train_set)*rate)]
    # tmp=-similarity_martix[train_set[:,0].astype('int'),train_set[:,1].astype('int')];tmp.sort()
    # print(tmp) #debug
    return train_set[thresholded_align]


def predict(model, output_file, sim_measure="cosine", save_emb=False, save_sim=False):
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
    if save_emb:
        if not isinstance(Embedding1, np.ndarray):
            Embedding1 = Embedding1.numpy()
        if not isinstance(Embedding2, np.ndarray):
            Embedding2 = Embedding2.numpy()
        np.save(output_file.replace('.txt', f'_emb1.npy'), Embedding1)
        np.save(output_file.replace('.txt', f'_emb2.npy'), Embedding2)
    if save_sim:
        np.save(output_file.replace('.txt', f'_sim.npy'), similarity_matrix)


def evaluate(model, data, k, sim_measure="cosine", phase="test", add_trust=False, trust_set=None):
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
        gt = int(data[idx][1])  # todo
        if int(gt) == alignment_hit1[g1_idx]:
            hit_1_score += 1
            if add_trust and isinstance(trust_set, set):
                trust_set.add((g1_idx, gt))
        if int(gt) in alignment_hitk[g1_idx]:
            hit_k_score += 1
    return similarity_matrix, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score


def train(model, optimizer, scheduler, train_set, test_set, train_loader, base_rate=0.2, max_rate=0.8, rateiter=100):
    # begin training
    best_E1 = None
    best_E2 = None
    best_hit_1_score = 0
    best_hit_1_epoch = 0
    # best_alignment=None
    rate = base_rate
    train_set_ori=train_set.copy()
    neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(train_set, neg_samples_size=neg_samples_size)
    for e in range(epoch):
        model.train()
        if e != 0 and e % rateiter == 0: # no threashold at beginning
            rate = min(2 * rate, max_rate)
            # change train_set
            E1, E2 = model()
            train_set=generate_train_set_with_threashold(train_set_ori,E1,E2,rate=rate)
            print('@len train set',train_set.shape)
            neg1_left, neg1_right, neg2_left, neg2_right = generate_neg_sample(train_set, neg_samples_size=neg_samples_size)
            train_loader = DataLoader(dataset=Dataset(train_set), batch_size=len(train_set), shuffle=False)
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
            sim_mat, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score = evaluate(model, data=test_set, k=k)

            if hit_1_score > best_hit_1_score:
                best_hit_1_score = hit_1_score
                # best_alignment=alignment_hit1
                best_hit_1_epoch = e
                torch.save(model, 'model_best.pth')
                # todo save model
                logging.info(
                    f"current best Hits@1 count {hit_1_score}({round(hit_1_score/len(test_set), 2)}) ,hit@{k}: total {hit_k_score}({round(hit_k_score/len(test_set), 2)}) at the {e+1}th epoch: {best_hit_1_score}")

        tb_logger.add_scalar('loss_train', loss.item(), e + 1)
        tb_logger.add_scalar('hit1_rate', hit_1_score / len(test_set), e + 1)
        logging.info(f"epoch: {e+1}, loss: {round(loss.item(), 3)}\n")
    logging.info(
        f"@train finished. current best Hits@1 count {best_hit_1_score}({round(best_hit_1_score/len(test_set), 2)}) ,at the {best_hit_1_epoch}th epoch")
    # return best_alignment
# final evaluation and test
# ground_truth = np.loadtxt('ground_truth.txt', delimiter=' ')
# ground_truth = np.loadtxt('data/anchor/anchor_0.2_test.txt', delimiter=' ')
# similarity_matrix, alignment_hit1, alignment_hitk, hit_1_score, hit_k_score = evaluate(data=ground_truth, k=k, phase="over")
# logging.info(similarity_matrix)
# logging.info(f"final score: hit@1: total {hit_1_score} and ratio {round(hit_1_score/len(ground_truth), 2)}, hit@{k}: total {hit_k_score} and ratio {round(hit_k_score/len(ground_truth), 2)}")


if __name__ == '__main__':

    os.makedirs('kfoldCV', exist_ok=True)
    import numpy as np

    kf = KFold(n_splits=args.cv)
    kf.get_n_splits(anchor)

    logging.info(f'@kf = {args.cv}')
    fold = 0
    logging.info('=' * 40 + f'fold {fold+1}' + '=' * 40)

    # diff tb
    board_path = args.board_path + '/' + EXP_NAME + f'_fold{fold}'
    os.makedirs(board_path, exist_ok=True)
    tb_logger = SummaryWriter(board_path)
    # dataset
    train_set = np.loadtxt('pseudo_label.txt', delimiter=' ', dtype='int')
    test_set = anchor
    train_set, test_set = np.array(list(train_set)), np.array(list(test_set))
    train_size, test_size = len(train_set), len(test_set)
    batchsize = len(train_set)
    train_dataset = Dataset(train_set)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=False)
    model = Model(Variable(torch.from_numpy(A1).float()), Variable(torch.from_numpy(A2).float()), embedding_dim=embedding_dim)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_adjust_freq, args.lr_decay_rate)
    logging.info(f"training samples: {train_size}, test samples: {test_size}")
    if fold == 0:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f'#total params: {pytorch_total_params}')
        logging.info(f"model architecture:\n {model}")

    train(model=model, optimizer=optimizer, scheduler=scheduler, train_set=train_set, test_set=test_set, train_loader=train_loader)
    # 预测写入文件+emb写入文件
    model = torch.load('model_best.pth')
    predict(model, f'kfoldCV/submit_tmp_{args.graph_s}_{args.graph_d}_{anoise}+fold{fold}.txt', save_sim=True)
