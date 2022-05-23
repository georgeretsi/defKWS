############ config #############

import os

dataset_path = './saved_datasets' # path to save the process dataset for quick loading
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)
save_path = './saved_models/' # path for saving models
if not os.path.isdir(save_path):
    os.mkdir(save_path)

######################################

dataset='IAM'

fixed_size = (64, 256)
max_epochs = 240
batch_size = 64
display=100

# ctc auxiliary loss
ctc_aux = True

######################################

classes = '_*0123456789abcdefghijklmnopqrstuvwxyz '

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}

def reduced(istr):
    return ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in istr.lower()])

######################################

from utils.phoc import build_phoc_descriptor
levels = [1, 2, 3, 4]
phoc = lambda x: build_phoc_descriptor(x, classes, levels)
phoc_size = sum(levels) * len(classes)

######################################

# architecture configuration
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)]


######################################

load_code = 'phoc_kws_a_iam'
name_code = 'phoc_kws_a'
name_code += '_iam'

######################################
from iam_config import stopwords_path
stopwords = []
if dataset=='IAM':
    for line in open(stopwords_path):
        stopwords.append(line.strip().split(','))
    stopwords = stopwords[0]
