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

from utils.auxilary_functions import average_precision

from config import *

logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('HTR-Experiment::train')
logger.info('--- Running HTR Training ---')
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

val_set = IAMLoader('./saved_datasets', 'val', level='word', fixed_size=fixed_size)
test_set = IAMLoader('./saved_datasets', 'test', level='word', fixed_size=fixed_size)

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


optimizer = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5*max_epochs), int(.75*max_epochs)])


ctc_loss = lambda y, t, ly, lt: nn.CTCLoss(reduction='mean', zero_infinity=True)(F.log_softmax(y, dim=2), t, ly, lt)

bce_loss = nn.BCEWithLogitsLoss()


def test_kws_adaptive_qbe(epoch, cutoff=50, iters=3):

    net.eval()

    logger.info('Testing KWS at epoch %d', epoch)
    tdecs = []
    ctc_feats = []
    transcrs = []

    for idx, (img, transcr) in enumerate(test_loader):
        img = Variable(img.cuda(gpu_id))
        with torch.no_grad():
            ctc_tmp, feat = net(img)

            #tdec = F.normalize(feat.sigmoid(), dim=1)
            ctc_feats += [ctc_tmp.permute(1, 0 , 2)]
            tdec = feat #.sigmoid()

        tdecs += [tdec]
        transcrs += [reduced(t.strip()) for t in list(transcr)]
    tdecs = torch.cat(tdecs)
    ctc_feats = torch.cat(ctc_feats)
    #print(tdecs)

    uwords = np.unique(transcrs)
    udict = {w: i for i, w in enumerate(uwords)}
    lbls = np.asarray([udict[w] for w in transcrs])
    cnts = np.bincount(lbls)

    queries = [w for w in uwords if w not in stopwords and cnts[udict[w]] > 1 and len(w) > 1 and '*' not in w]

    qids = np.asarray([i for i,t in enumerate(transcrs) if t in queries])

    qdecs = tdecs[qids]

    D = 1 - torch.mm(qdecs, tdecs.t()) / (qdecs.pow(2).sum(dim=-1, keepdim=True).sqrt() * tdecs.pow(2).sum(dim=-1, keepdim=True).t().sqrt())

    D = D.cpu().numpy()
    Id = np.argsort(D, axis=1)
    while Id.max() > Id.shape[1]:
        Id = np.argsort(D, axis=1)

    map_qbe = 100 * np.mean([average_precision(lbls[Id[i]][1:] == lbls[qc]) for i, qc in enumerate(qids)])
    logger.info('Initial QBE MAP at epoch %d: %f', epoch, map_qbe)

    # recompute distances based on the adaptive deformation scheme
    for i, qid in enumerate(qids):

        tids = Id[i, 1:cutoff]
        imgs = torch.stack([test_set.__getitem__(tid)[0] for tid in tids], 0)
        imgs = Variable(imgs.cuda(gpu_id))
        target = (tdecs[qid].view(1, -1).repeat(imgs.size(0), 1),
                  tdecs[tids])
        timg = test_set.__getitem__(Id[i, 0])[0].cuda(gpu_id).unsqueeze(0).repeat(imgs.size(0), 1, 1, 1)
        vals = D[i, tids]
        for rr in range(1):
            tvals = img_deformation(imgs, timg, target, c=1., lr=.01, N=iters, laffine=True, gaffine=True, ldeform=True)[1]
            vals = np.minimum(vals, tvals.cpu().numpy())

        D[i, tids] = vals #vals.cpu().numpy()

    Id = np.argsort(D, axis=1)
    while Id.max() > Id.shape[1]:
        Id = np.argsort(D, axis=1)

    map_qbe = 100 * np.mean([average_precision(lbls[Id[i]][1:] == lbls[qc]) for i, qc in enumerate(qids)])
    logger.info('Adaptive QBE MAP at epoch %d: %f', epoch, map_qbe)


def local_affine(img, aff, wsize=32, wstride=16):

    Ksegs = int(1 + (img.size(3) - wsize)/wstride)

    if aff.size(0) != Ksegs * img.size(0):
        print("number of local affine matrices should correspond to kernel/stride info")

    ref_aff = torch.Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).unsqueeze(0).to(img.device)

    ref_grid = F.affine_grid(ref_aff, (1, 1, img.size(2), wsize))
    local_grids = F.affine_grid((aff + ref_aff).view(-1, 2, 3), (img.size(0) * Ksegs, img.size(1), img.size(2), wsize))
    local_grids = local_grids.view(Ksegs, img.size(0), img.size(2), wsize, 2) - ref_grid.unsqueeze(0)


    ww = torch.zeros(Ksegs, img.size(3), device=img.device)
    for i in range(Ksegs):
        ww[i, i * wstride:i *wstride + wsize] = torch.cat([torch.linspace(0, 1, wsize//2), torch.linspace(1, 0, wsize//2)]).to(img.device)
    ww /= (ww.sum(dim=0) + 1e-5)

    # add local offset
    for i in range(Ksegs):
        local_grids[i] *= ww[i, i * wstride: i * wstride + wsize].view(1, 1, -1, 1)

    dgrid = F.fold(local_grids.permute(1, 4, 2, 3, 0).reshape(img.size(0), -1, Ksegs),
                  kernel_size=(img.size(2), wsize), stride=(img.size(2), wstride),
                  output_size=(img.size(2), img.size(3))).permute(0, 2, 3, 1)
    return dgrid


def img_deformation(img, timg, target, c=1.0, lr=1e-3, N=25, gaffine=True, laffine=True, ldeform=False):

    parameter_list = []

    aff_constraint = torch.tensor([[0.2, 0.2, 0.25], [0.2, 0.2, 0.25]], device=img.device).unsqueeze(0)
    ref_aff = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=img.device).unsqueeze(0)

    if laffine:
        laff_wsize, laff_wstride = 16, 8
        Ksegs = int(1 + (img.size(3) - laff_wsize) / laff_wstride)
        laff = .001 * torch.randn((Ksegs*img.size(0), 2, 3), device=img.device)
        laff = nn.Parameter(laff)
        parameter_list += [laff]

    if gaffine:
        aff = .001 * torch.rand((img.size(0), 2 ,3)).to(img.device)
        aff = nn.Parameter(aff)
        parameter_list += [aff]

    if ldeform:
        ld = nn.Parameter(.01 * torch.rand(img.size(0), 2, img.size(2) // 8, img.size(3) // 8).to(img.device))
        parameter_list += [ld]

    toptimizer = torch.optim.Adam(parameter_list, lr)

    tdec, rdec = target

    floss = torch.zeros(img.size(0), device=img.device)

    for i in range(N):
        toptimizer.zero_grad()

        nimg = img
        eloss = 0.0


        if gaffine:
            naff = aff_constraint * aff.tanh()
            nrm = F.affine_grid(ref_aff + naff, img.size())
        else:
            nrm = F.affine_grid(ref_aff.repeat(img.size(0), 1, 1), img.size())

        if laffine:
            nlaff = aff_constraint * laff.tanh()
            eloss += laff.norm()
            nrm += local_affine(nimg, nlaff, laff_wsize, laff_wstride)

        if ldeform:
            ldv = .1 * F.interpolate(ld, (img.size(2), img.size(3))) #.tanh()
            eloss += ld.norm()
            nrm += ldv.permute(0, 2, 3, 1)

        nimg = F.grid_sample(img, nrm, padding_mode='border')

        seq, dec = net(nimg)

        cos_loss = 1 - F.cosine_similarity(dec, tdec).mean()

        floss +=1 - F.cosine_similarity(dec, rdec).detach()

        rdec = dec.detach()

        loss_val = 10.0 * F.mse_loss(nimg, timg) + 1 * cos_loss + c * eloss
        #loss_val = 10 * cos_loss + c * eloss

        loss_val.backward()

        toptimizer.step()

    fimg = img
    if gaffine:
        naff = aff_constraint * aff.tanh()
        nrm = F.affine_grid(ref_aff + naff, img.size())
    else:
        nrm = F.affine_grid(ref_aff.repeat(img.size(0), 1, 1), img.size())

    if laffine:
        nlaff = aff_constraint * laff.tanh()
        nrm += local_affine(fimg, nlaff, laff_wsize, laff_wstride)

    if ldeform:
        ldv = .1 * F.interpolate(ld, (img.size(2), img.size(3)))
        nrm += ldv.permute(0, 2, 3, 1)

    fimg = F.grid_sample(fimg, nrm, padding_mode='border')

    with torch.no_grad():
        dec = net(fimg)[1] #.sigmoid()
    floss = 1 - F.cosine_similarity(dec, tdec)

    return fimg, floss

test_kws_adaptive_qbe(0)