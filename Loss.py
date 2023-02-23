import torch
import torch.nn.functional as F
from torch.nn import Parameter

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0.1):
    # Optional: BNInception uses label smoothing, apply it for retraining also
    # "Rethinking the Inception Architecture for Computer Vision", p. 6
    # import sklearn.preprocessing
    T = T.cpu().numpy()
    # T = sklearn.preprocessing.label_binarize(
    #     T, classes = range(0, nb_classes)
    # )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()
    return T


class CMSP_out(torch.nn.Module):
    def __init__(self, nb_classes, sz_embedding, mrg, alpha, beta, gamma, **kwargs):
        torch.nn.Module.__init__(self)
        self.proxies = torch.nn.Parameter((torch.randn(nb_classes, sz_embedding)/8).cuda())
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mrg = mrg
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scaling_x = 1
        self.scaling_p = 3
        self.smoothing_const = 0.1

    def forward(self, feature_1, feature_2,  predict_1, predict_2, label_1, label_2):
        feature_all = torch.cat((feature_1, feature_2), dim=0)
        # predict_all = torch.cat((predict_1, predict_2), dim=0)
        label_all = torch.cat((label_1, label_2), dim=0)
        # proxies = self.proxies
        proxies = F.normalize(self.proxies, p=2, dim=-1) * self.scaling_x
        feature_all = F.normalize(feature_all, p=2, dim=-1) * self.scaling_p
        

        D_ = torch.cdist(feature_all, proxies)**2

        mrg = torch.zeros_like(D_)
        mrg[label_all == 1] = mrg[label_all == 1] + self.mrg
        D_ = D_ + mrg

        label_all = binarize_and_smooth_labels(label_all, len(proxies), self.smoothing_const)
        p_loss = torch.sum(-label_all * F.log_softmax(-D_, 1), -1).mean()
        # p_loss = p_loss.mean()
        d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1)) + \
                 self.cross_entropy(predict_2, torch.argmax(label_2, -1))
        # d_loss = ((predict_1 - label_1.float()) ** 2).sum(1).sqrt().mean() + (
        #             (predict_2 - label_2.float()) ** 2).sum(1).sqrt().mean()
        # m_loss = F.mse_loss(feature_1, feature_2)
        m_loss = ((feature_1 - feature_2) ** 2).sum(1).sqrt().mean()

        loss = self.alpha * p_loss + self.beta * d_loss + self.gamma * m_loss
        return loss


# class CMSP_in(torch.nn.Module):
#     def __init__(self, nb_classes, sz_embedding, mrg, alpha, beta, gamma, **kwargs):
#         torch.nn.Module.__init__(self)
#         self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embedding).cuda())
#         self.cross_entropy = torch.nn.CrossEntropyLoss()
#         self.mrg = mrg
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.n_view = 2

#     def forward(self, feature_1, feature_2,  predict_1, predict_2, label_1, label_2):
#         # feature_all = torch.cat((feature_1, feature_2), dim=0)
#         # # predict_all = torch.cat((predict_1, predict_2), dim=0)
#         # label_all = torch.cat((label_1, label_2), dim=0)
#         # proxies = self.proxies

#         proxies = F.normalize(self.proxies, p=2, dim=-1)
#         # feature_all = F.normalize(feature_all, p=2, dim=-1)
#         feature_1 = F.normalize(feature_1, p=2, dim=-1)
#         feature_2 = F.normalize(feature_2, p=2, dim=-1)

#         # D = torch.cdist(feature_all, proxies) ** 2
#         D_1 = torch.cdist(feature_1, proxies)
#         D_2 = torch.cdist(feature_2, proxies)

#         p_loss = torch.sum(-label_1 * torch.log((1/self.n_view)*(F.softmax(-D_1, 1)+F.softmax(-D_2, 1))), -1).mean()
#         # p_loss = p_loss.mean()
#         d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1)) + \
#                  self.cross_entropy(predict_2, torch.argmax(label_2, -1))
#         # d_loss = ((predict_1 - label_1.float()) ** 2).sum(1).sqrt().mean() + (
#         #             (predict_2 - label_2.float()) ** 2).sum(1).sqrt().mean()
#         # m_loss = F.mse_loss(feature_1, feature_2)
#         m_loss = ((feature_1 - feature_2) ** 2).sum(1).sqrt().mean()

#         loss = self.alpha * p_loss + self.beta * d_loss + self.gamma * m_loss
#         return loss


# class P_loss_cos(torch.nn.Module):
#     def __init__(self, nb_classes, sz_embedding, mrg, alpha, beta, gamma, **kwargs):
#         torch.nn.Module.__init__(self)
#         self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embedding).cuda())
#         self.cross_entropy = torch.nn.CrossEntropyLoss()
#         self.mrg = mrg
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#     def forward(self, feature_1, feature_2,  predict_1, predict_2, label_1, label_2):
#         feature_all = torch.cat((feature_1, feature_2), dim=0)
#         # predict_all = torch.cat((predict_1, predict_2), dim=0)

#         label_all = torch.cat((label_1, label_2), dim=0)

#         # proxies = F.normalize(self.proxies, p=2, dim=-1)
#         # feature_all = F.normalize(feature_all, p=2, dim=-1)
#         proxies = self.proxies

#         D = F.linear(self.l2_norm(feature_all), self.l2_norm(proxies))
#         D[label_all == 1] += self.mrg

#         p_loss = torch.sum(-label_all * F.log_softmax(D, 1), -1).mean()

#         d_loss = self.cross_entropy(predict_1, torch.argmax(label_1, -1)) + \
#                  self.cross_entropy(predict_2, torch.argmax(label_2, -1))

#         m_loss = ((feature_1 - feature_2) ** 2).sum(1).sqrt().mean()

#         loss = self.alpha * p_loss + self.beta * d_loss + self.gamma * m_loss
#         return loss