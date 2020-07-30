import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import math
from matplotlib import gridspec

sys.path.append('../network')
from cnn import CNN

def normalise_img(img):
    img = img - img.min()
    img = img / img.max()
    return img * 255

if __name__ == '__main__':

    model_name = 'bs_64_lr_0.001_run_85'
    model_path = f'../logs/{model_name}/model.pt'

    resize_transform = transforms.Resize((360, 640))
    tensor_transform = transforms.ToTensor()

    image_path = f'../../train/0a0c3694-4cc8b0e3/0a0c3694-4cc8b0e3-8.png'

    image = Image.open(image_path)

    image_tensor = tensor_transform(resize_transform(image)).unsqueeze(0)

    model = CNN(640, 360, 3)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Visualize feature maps
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))
    model.conv4.register_forward_hook(get_activation('conv4'))
    model.conv5.register_forward_hook(get_activation('conv5'))

    output = model(image_tensor)

    for layer, act in activation.items():
        act = act.squeeze()

        im_width = act[0].shape[1]
        im_height = act[0].shape[0]

        nrows = 4
        ncols = act.size(0) // nrows

        new_im = Image.new('L', (ncols*im_width, nrows*im_height))

        for row in range(nrows):
            for col in range(ncols):
                idx = row*ncols + col

                image = act[idx]

                im = Image.fromarray(normalise_img(image.numpy()))

                new_im.paste(im, (col*im_width,row*im_height))

        new_im.save(f'images/activation-{layer}.png', format='png', resolution=400)