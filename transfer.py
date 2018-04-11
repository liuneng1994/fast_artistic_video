import argparse
import torch
from imageio import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

parser = argparse.ArgumentParser("图像风格迁移参数")
parser.add_argument("--image_path", default="data/images/u=2257830751,2582425102&fm=27&gp=0.jpg")
parser.add_argument("--image_size", default=128)
parser.add_argument("--model_path", default='model.ckpt')
args = parser.parse_args()


def main():
    model = torch.load(args.model_path).cuda()
    origin_image = imread(args.image_path)
    image = resize(origin_image, (args.image_size, args.image_size))
    origin_size = origin_image.shape
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    undo_preprocess = transforms.Compose([
        #transforms.Normalize(mean=[-0.485, -0.456, -0.406],
         #                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage()])

    image = preprocess(image)
    styled = model(torch.unsqueeze(image, 0).cuda()).squeeze()
    styled_image = undo_preprocess(styled.data.cpu())
    styled_image = resize(np.array(styled_image), (origin_size[0], origin_size[1]))
    plt.subplot(2, 1, 1)
    plt.imshow(origin_image)
    plt.subplot(2, 1, 2)
    plt.imshow(styled_image)
    plt.show()


if __name__ == '__main__':
    main()
