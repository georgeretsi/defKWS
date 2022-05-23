import argparse
import logging

import os
import numpy as np
import torch.cuda
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from iam_loader import IAMLoader

from models import KWSNet

import torch.nn.functional as F

from utils.auxilary_functions import affine_transformation

from config import *

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('KWS-Experiment::train')
logger.info('--- Running KWS Training ---')
# argument parsing
parser = argparse.ArgumentParser()
# - train arguments

parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                    help='The ID of the GPU to use. If not specified, training is run in CPU mode.')

args = parser.parse_args()

gpu_id = args.gpu_id

# print out the used arguments
logger.info('###########################################')

logger.info('Loading dataset.')

aug_transforms = [lambda x: affine_transformation(x, s=.2)]

train_set = IAMLoader('./saved_datasets', 'train', level='word', fixed_size=fixed_size, transforms=aug_transforms)
val_set = IAMLoader('./saved_datasets', 'val', level='word', fixed_size=fixed_size)
test_set = IAMLoader('./saved_datasets', 'test', level='word', fixed_size=fixed_size)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8) #, drop_last=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

######################################

net = KWSNet(cnn_cfg, phoc_size, len(classes))

if load_code is not None:
    net.load_my_state_dict(torch.load(save_path + load_code + '.pt'))

net.cuda(args.gpu_id)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Network ready. #paramaters: " + str(count_parameters(net)))

######################################

optimizer = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])

ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='mean', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt)

bce_loss = nn.BCEWithLogitsLoss()


def train(epoch):

    net.train()

    closs = []
    for iter_idx, (img, transcr) in enumerate(train_loader):

        transcr = [reduced(t) for t in transcr]
        phoc_gt = torch.Tensor(phoc(transcr)).cuda(gpu_id)

        optimizer.zero_grad()

        img = Variable(img.cuda(gpu_id))
        output, phoc_est = net(img)

        if ctc_aux:
            labels = torch.IntTensor([cdict[c] for c in ''.join(transcr)])  # .to(img.device)
            label_lens = torch.IntTensor([len(t) for t in transcr])  # .to(img.device)

            act_lens = torch.IntTensor(img.size(0) * [output.size(0)])

            loss_val = ctc_loss(output.cpu(), labels, act_lens, label_lens) + \
                           100 * bce_loss(phoc_est, phoc_gt).cpu()
        else:
            loss_val = 1. * bce_loss(phoc_est, phoc_gt)


        closs += [loss_val.data]

        loss_val.backward()
        optimizer.step()

        if iter_idx % display == display-1:
            logger.info('Epoch %d, Iteration %d: %f', epoch, iter_idx+1, sum(closs)/len(closs))
            closs = []


from utils.auxilary_functions import average_precision

def test_kws(epoch):

    net.eval()

    logger.info('Testing KWS at epoch %d', epoch)
    tdecs = []
    transcrs = []
    for (img, transcr) in test_loader:

        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            _, feat = net(img)

            tdec = F.normalize(feat, dim=1).cpu().numpy().squeeze()

        tdecs += [tdec]
        transcrs += [reduced(t) for t in transcr]

    tdecs = np.concatenate(tdecs)

    # find unique words : wuary candidates
    uwords = np.unique(transcrs)
    udict = {w: i for i, w in enumerate(uwords)}
    lbls = np.asarray([udict[w] for w in transcrs])
    cnts = np.bincount(lbls)

    # keep as queries word with more than 1 instances, not int the stopword list and without any special character
    queries = [w for w in uwords if w not in stopwords and cnts[udict[w]] > 1 and len(w) > 1 and '*' not in w]


    qids = np.asarray([i for i,t in enumerate(transcrs) if t in queries])
    qdecs = tdecs[qids]

    D = -np.dot(qdecs, np.transpose(tdecs))

    Id = np.argsort(D, axis=1)
    while Id.max() > Id.shape[1]:
        Id = np.argsort(D, axis=1)

    map_qbe = 100 * np.mean([average_precision(lbls[Id[i]][1:] == lbls[qc]) for i, qc in enumerate(qids)])
    logger.info('QBE MAP at epoch %d: %f', epoch, map_qbe)

    net.train()


test_kws(0)
for epoch in range(1, max_epochs + 1):

    train(epoch)

    scheduler.step()

    if epoch % 5 == 0:
        test_kws(epoch)

    if epoch % 10 == 0:
        logger.info('Saving net after %d epochs', epoch)
        torch.save(net.cpu().state_dict(), save_path+name_code+'.pt')
        net.cuda(gpu_id)
