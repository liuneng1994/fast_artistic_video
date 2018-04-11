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
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

# TODO 实现数据集读取

parser = argparse.ArgumentParser(description="视频风格迁移训练参数配置")

parser.add_argument("--style_image", default='data/shuimo.jpg')
parser.add_argument("--image_size", default=128)

parser.add_argument("--use_instance_norm", default=0)
parser.add_argument("--padding_type", default='reflect')
parser.add_argument("--tanh_constant", default=150)
parser.add_argument("--tv_strength", default=1e-6)
parser.add_argument("--content_weights", default=1.0)
parser.add_argument("--style_weights", default=1e3)

# Optimization
parser.add_argument("--num_iterations", default=40000)
parser.add_argument("--batch_size", default=4)
parser.add_argument("--learning_rate", default=1e-3)
args = parser.parse_args()


def main():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    style_image = tf(imresize(imread(args.style_image), (args.image_size, args.image_size)))
    style_image = torch.unsqueeze(style_image, 0)
    dataset = DataLoader(mscocoDataset(file='data/ms-coco-256.h5'), batch_size=args.batch_size, shuffle=False,
                         num_workers=4)
    net = ArtisticNet(args)
    net.build_model()
    if cuda.is_available():
        net = net.cuda()
    cnn = models.vgg19(pretrained=True).features
    optimizer = optim.Adam(lr=args.learning_rate, params=net.parameters())
    if cuda.is_available():
        cnn = cnn.cuda()
    step = 0

    writer = SummaryWriter('./logs')

    def auto_save():
        if step % 200 == 0:
            torch.save(net, "model.ckpt")

    def log_gradient(parameters, step):
        for name, parameter in parameters:
            writer.add_histogram(name, parameter.grad.data.cpu().numpy(), step, bins='auto')

    for param in net.parameters():
        if len(param.data.size()) < 2:
            nn.init.constant(param.data, 0)
        else:
            nn.init.xavier_normal(param.data)
    while step < args.num_iterations:
        for data in dataset:
            step += 1
            if step >= args.num_iterations:
                break
            auto_save()
            x = autograd.Variable(data).cuda() if cuda.is_available() else autograd.Variable(data)
            styled_image = net(x)
            target = autograd.Variable(style_image)
            if cuda.is_available():
                target = target.cuda()
            loss = PerceptualLoss(cnn, target, x)

            def closure():
                net.zero_grad()
                losses, cl, sl, tl = loss(styled_image)
                losses.backward()
                # nn.utils.clip_grad_norm(net.parameters(), 1000)
                if step % 10 == 0:
                    print("step %d loss %f cl %f sl %f tl %f" % (step, losses, cl, sl, tl))
                    writer.add_scalars("LOSS", {"content_loss": cl, "style_loss": sl, "tv_loss": tl, "loss": losses},
                                       global_step=step)
                    writer.add_image("style_image", make_grid(styled_image.data), step)
                    log_gradient(net.named_parameters(), step)
                return losses

            optimizer.step(closure=closure)
    torch.save(net, "model.ckpt")


if __name__ == '__main__':
    main()
