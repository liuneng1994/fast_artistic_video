import torch
from torch import nn, autograd, cuda
from model.GramMatrix import GramMatrix
from model.ContentLoss import ContentLoss
from model.StyleLoss import StyleLoss
from model.TVLoss import TVLoss


class PerceptualLoss(nn.Module):
    def __init__(self, cnn, style_img, content_img,
                 style_weight=100,
                 content_weight=10,
                 tv_loss_weight=1e-4,
                 content_layers=('conv_4'),
                 style_layers=('conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5')
                 ):
        super(PerceptualLoss, self).__init__()
        self.cnn = cnn
        self.style_img = style_img
        self.content_img = content_img
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.model = nn.Sequential()
        self.gram = GramMatrix()
        self.content_losses = []
        self.style_losses = []
        self.tv_loss = TVLoss(tv_loss_weight)
        if cuda.is_available():
            self.model = self.model.cuda()
            self.gram = self.gram.cuda()
        self.init()

    def init(self):
        i = 1
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                self.model.add_module(name, layer)

                if name in self.content_layers:
                    # add content loss:
                    target = self.model(self.content_img).clone()
                    content_loss = ContentLoss(target, self.content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in self.style_layers:
                    # add style loss:
                    target_feature = self.model(self.style_img).clone()
                    target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, self.style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                self.model.add_module(name, layer)

                if name in self.content_layers:
                    # add content loss:
                    target = self.model(self.content_img).clone()
                    content_loss = ContentLoss(target, self.content_weight)
                    self.model.add_module("content_loss_" + str(i), content_loss)
                    self.content_losses.append(content_loss)

                if name in self.style_layers:
                    # add style loss:
                    target_feature = self.model(self.style_img).clone()
                    target_feature_gram = self.gram(target_feature)
                    style_loss = StyleLoss(target_feature_gram, self.style_weight)
                    self.model.add_module("style_loss_" + str(i), style_loss)
                    self.style_losses.append(style_loss)

                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                self.model.add_module(name, layer)  # ***

    def forward(self, input):
        self.model(input)
        loss = None
        closs = None
        sloss = None
        for cl in self.content_losses:
            if loss is None:
                loss = cl.loss
                closs = cl.loss.clone()
            else:
                loss += cl.loss
                closs += cl.loss
        for sl in self.style_losses:

            if loss is None:
                loss = sl.loss * (1 / len(self.style_losses))
            else:
                loss += sl.loss * (1 / len(self.style_losses))
                if sloss is None:
                    sloss = sl.loss * (1 / len(self.style_losses))
                else:
                    sloss += sl.loss * (1 / len(self.style_losses))
        tloss = self.tv_loss(input)
        loss += tloss

        return loss, closs, sloss, tloss
