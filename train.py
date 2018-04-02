import torch
from torch import nn, cuda, autograd, optim
from model.PerceptualLoss import PerceptualLoss
from model.ArtisticNet import ArtisticNet
import argparse
import torchvision.models as models
from torch.utils.data.dataloader import DataLoader
from dataset.mscoco_dataset import mscocoDataset
from scipy.misc import imread, imresize
import torchvision.transforms as transforms

# TODO 实现数据集读取

parser = argparse.ArgumentParser(description="视频风格迁移训练参数配置")

parser.add_argument("--style_image", default='data/shuimo.jpg')

parser.add_argument("--use_instance_norm", default=1)
parser.add_argument("--padding_type", default='reflect')
parser.add_argument("--tanh_constant", default=150)
parser.add_argument("--tv_strength", default=1e-6)
parser.add_argument("--content_weights", default=1.0)
parser.add_argument("--style_weights", default=5.0)

# Optimization
parser.add_argument("--num_iterations", default=40000)
parser.add_argument("--batch_size", default=1)
parser.add_argument("--learning_rate", default=1e-3)
args = parser.parse_args()


def main():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    style_image = tf(imresize(imread(args.style_image), (256, 256)))
    style_image = torch.unsqueeze(style_image, 0)
    dataset = DataLoader(mscocoDataset(file='data/ms-coco-256.h5'), batch_size=args.batch_size, shuffle=True,
                         num_workers=4)
    net = ArtisticNet(args)
    net.build_model()
    cnn = models.vgg19(pretrained=True).features
    optimizer = optim.Adam(lr=args.learning_rate, params=net.parameters())
    if cuda.is_available():
        cnn = cnn.cuda()
    step = 0

    def auto_save():
        if step % 10000 == 0:
            torch.save(net, "model.ckpt")

    while step < args.num_iterations:
        for data in dataset:
            step += 1
            if step >= args.num_iterations:
                break
            auto_save()
            x = autograd.Variable(data)
            styled_image = net(x)
            loss = PerceptualLoss(cnn, autograd.Variable(style_image), x)

            def closure():
                net.zero_grad()
                losses = loss(styled_image)
                losses.backward()
                # if step % 100 == 0:
                print("step %d loss %f" % (step, losses))
                return losses

            optimizer.step(closure=closure)


if __name__ == '__main__':
    main()
