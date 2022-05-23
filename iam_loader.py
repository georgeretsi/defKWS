'''
@author: georgeretsi
'''

import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset

from os.path import isfile

from utils.auxilary_functions import image_resize, centered

from iam_config import *

def gather_iam_info(set='train', level='word'):

    # train/test file
    if set == 'train':
        valid_set = np.loadtxt(trainset_file, dtype=str)
    elif set == 'test':
        valid_set = np.loadtxt(testset_file, dtype=str)
    elif set == 'val':
        valid_set = np.loadtxt(valset_file, dtype=str)
    else:
        print('split name not found. Valid splits: [train, val, test]')
        return


    if level == 'word':
        gtfile= word_file
        root_path = word_path
    elif level == 'line':
        gtfile = line_file
        root_path = line_path
    else:
        print('segmentation level not found. Valid segmentation level: [word, line[')
        return

    gt = []
    for line in open(gtfile):
        if not line.startswith("#"):
            info = line.strip().split()
            name = info[0]

            name_parts = name.split('-')
            pathlist = [root_path] + ['-'.join(name_parts[:i+1]) for i in range(len(name_parts))]
            if level == 'word':
                tname = pathlist[-2]
                del pathlist[-2]
            elif level == 'line':
                tname = pathlist[-1]

            if (info[1] != 'ok') or (tname not in valid_set):
            #if (line_name not in valid_set):
                continue

            img_path = '/'.join(pathlist)

            transcr = ' '.join(info[8:])
            gt.append((img_path, transcr))

    return gt

def main_loader(set, level):

    info = gather_iam_info(set, level)

    data = []
    for i, (img_path, transcr) in enumerate(info):

        if i % 1000 == 0:
            print('imgs: [{}/{} ({:.0f}%)]'.format(i, len(info), 100. * i / len(info)))

        try:
            img = img_io.imread(img_path + '.png')
            img = 1 - img.astype(np.float32) / 255.0
            img = image_resize(img, height=img.shape[0] // 2)
        except:
            continue

        data += [(img, transcr.replace("|", " "))]

    return data

class IAMLoader(Dataset):

    def __init__(self, dataset_path, set, level='line', fixed_size=(128, None), transforms=None):

        self.transforms = transforms
        self.set = set
        self.fixed_size = fixed_size

        save_file = dataset_path + '/' + set + '_' + level + '.pt'

        if isfile(save_file) is False:
            data = main_loader(set=set, level=level)
            torch.save(data, save_file)
        else:
            data = torch.load(save_file)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index][0]

        transcr = self.data[index][1]
        fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

        if self.set == 'train':
            # random resize at training !!!
            nwidth = int(np.random.uniform(.5, 1.5) * img.shape[1])
            nheight = int((np.random.uniform(.8, 1.2) * img.shape[0] / img.shape[1]) * nwidth)
        else:
            nheight, nwidth = img.shape[0], img.shape[1]

        nheight, nwidth = max(4, min(fheight-16, nheight)), max(8, min(fwidth-32, nwidth))
        img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))

        img = centered(img, (fheight, fwidth), border_value=None)
        if self.transforms is not None:
            for tr in self.transforms:
                if np.random.rand() < .5:
                    img = tr(img)

        img = torch.Tensor(img).float().unsqueeze(0)

        return img, transcr
