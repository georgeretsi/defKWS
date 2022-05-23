import torch.nn as nn
import torch.nn.functional as F
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CNN(nn.Module):
    def __init__(self, cnn_cfg):
        super(CNN, self).__init__()

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, 2, 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_cfg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x))
                    in_channels = x
                    cnt += 1

    def forward(self, x):

        y = x
        for nn_module in self.features:
            y = nn_module(y)

        y = F.max_pool2d(y, [y.size(2), 1], stride=[y.size(2), 1])

        return y


class KWSNet(nn.Module):
    def __init__(self, cnn_cfg, phoc_size, nclasses=None):
        super(KWSNet, self).__init__()

        self.cnn = CNN(cnn_cfg)

        hidden_size = cnn_cfg[-1][-1]

        self.termporal = nn.Sequential(
            nn.Conv2d(hidden_size, nclasses, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
        )

        hidden_size = cnn_cfg[-1][-1]


        self.enc = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.1),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(hidden_size), nn.ReLU(), nn.Dropout(.1),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(hidden_size),
        )

        self.fnl = nn.Sequential(
            nn.ReLU(), nn.Dropout(.1),
            nn.Linear(4 * hidden_size, 4 * hidden_size), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(4 * hidden_size, phoc_size)
        )


    def forward(self, x):
        y = self.cnn(x)

        y_ctc = self.termporal(y)

        y_feat = self.enc(y)

        if self.training:
            return y_ctc.permute(2, 3, 0, 1)[0], self.fnl(y_feat.view(x.size(0), -1))
        else:
            return y_ctc.permute(2, 3, 0, 1)[0], y_feat.view(x.size(0), -1)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print("parameter missmatch @ " + name)