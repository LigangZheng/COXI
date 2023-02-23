import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    def __init__(self, dropout, input_dim=4096, output_dim=2048, pre_dim=10):
        super(ImgNN, self).__init__()
        mid_num = 2048
        self.denseL1 = nn.Linear(input_dim, mid_num)
        self.denseL2 = nn.Linear(mid_num, mid_num)
        self.denseL3 = nn.Linear(mid_num, output_dim)
        # self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        # out = self.dropout(F.relu(self.linear(x)))
        out = self.dropout(F.relu(self.denseL1(x)))
        out = self.dropout(F.relu(self.denseL2(out)))
        out = self.denseL3(F.relu(out))

        # out1 = self.linear(F.relu((out)))
        # norm_x = torch.norm(out, dim=1, keepdim=True)
        # out = out / norm_x
        return out


class TextNN(nn.Module):
    def __init__(self, dropout, input_dim=1024, output_dim=2048, pre_dim=10):
        super(TextNN, self).__init__()
        mid_num = 2048
        self.denseL1 = nn.Linear(input_dim, mid_num)
        self.denseL2 = nn.Linear(mid_num, mid_num)
        self.denseL3 = nn.Linear(mid_num, output_dim)
        # self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # out = self.dropout(F.relu(self.linear(x)))
        out = self.dropout(F.relu(self.denseL1(x)))
        out = self.dropout(F.relu(self.denseL2(out)))
        out = self.denseL3(F.relu(out))
        # out1 = self.linear(F.relu((out)))
        # norm_x = torch.norm(out, dim=1, keepdim=True)
        # out = out / norm_x
        return out


class IDCM_NN(nn.Module):
    def __init__(self, dropout, img_input_dim=4096, img_output_dim=1024,
                 text_input_dim=1024, text_output_dim=1024, feat_dim=512, pre_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(dropout, img_input_dim, img_output_dim, pre_dim)
        self.text_net = TextNN(dropout, text_input_dim, text_output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linearLayer = nn.Linear(img_output_dim, feat_dim)
        self.linearLayer2 = nn.Linear(feat_dim, pre_dim)
        self.relu = nn.ReLU()

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)

        # view1_feature = self.linearLayer(view1_feature)
        # view2_feature = self.linearLayer(view2_feature)
        view1_feature = self.dropout(self.linearLayer(view1_feature))
        view2_feature = self.dropout(self.linearLayer(view2_feature))

        view1_predict1 = self.linearLayer2(view1_feature)
        view2_predict2 = self.linearLayer2(view2_feature)

        return view1_feature, view2_feature, view1_predict1, view2_predict2
