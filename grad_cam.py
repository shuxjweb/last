import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import os
# import matplotlib.pyplot as plt

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):        # [8, 3, 256, 128]
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules.items():
            if name == 'features':    # [256, 3, 256, 128]
                x = module._modules['backbone'](x)          # [256, 2048, 16, 8]
                # t, att = module._modules['att'](x)        # [256, 2048, 16, 8], [256, 64, 16, 8]
                x.register_hook(self.save_gradient)
                outputs += [x]
                x = module._modules['avgpool'](x).squeeze()         # [256, 2048]
            else:
                x = module(x)       # [256, 625]
        return outputs, x    # [256, 64, 16, 8], [256, 625]


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):      # [8, 3, 256, 128]
        feat, output = self.feature_extractor(x)       # [256, 2048, 16, 8], [256, 625]
        return feat, output    # [256, 2048, 16, 8], [256, 625]


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask):           # [256, 128, 3], [256, 128]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)     # [256, 128, 3]
    heatmap = np.float32(heatmap) / 255     # [256, 128, 3]
    cam = heatmap + np.float32(img)         # [256, 128, 3]
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)               # [256, 128, 3]
    return cam



def print_images(seqs_n, att_n, home, id=0):           # [8, 256, 128, 3], [8, 64, 256, 128]
    b, c, h, w = att_n.shape

    for ii in range(c):
        img = seqs_n[id, ]       # [256, 128, 3]
        mask = att_n[id, ii]     # [256, 128]
        cam = show_cam_on_image(img, mask)     # [256, 128, 3]
        path_save = os.path.join(home, str(id) + '_' + str(ii) + '.jpg')
        cv2.imwrite(path_save, cam)
        path_save = os.path.join(home, str(id) + '_' + str(ii) + '_ini.jpg')
        cv2.imwrite(path_save, np.uint8(255 * img))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda, groups=64):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        self.groups = groups
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, seqs, index=None):     # [1, 8, 3, 256, 128]
        b, t, c, h, w = seqs.shape
        input = seqs.reshape((b * t, c, h, w))      # [8, 3, 256, 128]

        if self.cuda:
            features, output = self.extractor(input.cuda())   # [8, 2048, 16, 8], [8, 625]
        else:
            features, output = self.extractor(input)          # [8, 2048, 16, 8], [8, 625]
        features = features[0]       # [8, 2048, 16, 8]

        if True:
            output = output.reshape((b, t, -1))                    # [1, 8, 625]
            pool_output = torch.mean(output, dim=1)                # [1, 625]
        else:
            pool_output = output           # [8, 625]

        if index == None:
            index = np.argmax(pool_output.cpu().data.numpy(), axis=1)      # [1,]

        one_hot = np.zeros(pool_output.shape, dtype=np.float32)            # [1, 625]
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * pool_output)
        else:
            one_hot = torch.sum(one_hot * pool_output)    # 11.8605

        self.model.zero_grad()
        # one_hot.backward(retain_variables=True)
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()     # [8, 2048, 16, 8]
        target = features.cpu().data.numpy()           # [8, 2048, 16, 8]
        batch, channel, height, width = grads_val.shape

        num = int(channel / self.groups)
        grads_val = grads_val.reshape(batch, self.groups, -1, height, width)  # [8, 64, 32, 16, 8]
        target = target.reshape(batch, self.groups, -1, height, width)        # [8, 64, 32, 16, 8]

        weights = np.mean(grads_val, axis=(3, 4))        # [8, 64, 32]
        cam = np.zeros((batch, self.groups, height, width), dtype=np.float32)   # [8, 64, 16, 8]

        for ii in range(batch):
            for jj in range(self.groups):
                for kk in range(weights.shape[-1]):
                    cam[ii, jj] += weights[ii, jj, kk] * target[ii, jj, kk]

        att_n = np.zeros((batch, self.groups, h, w), dtype=np.float32)        # [8, 64, 256, 128]
        for ii in range(batch):
            for jj in range(self.groups):
                item = np.maximum(cam[ii, jj], 0)        # [16, 8]    if value < 0, set it 0
                item = cv2.resize(item, (w, h))          # [256, 128]
                item = item - np.min(item)
                item = item / (np.max(item) + 1e-5)      # [256, 128]
                att_n[ii, jj] = item
        return att_n


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward()

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=models.vgg19(pretrained=True), target_layer_names=["35"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)    # [224, 224, 3]
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)   # [1, 3, 224, 224]

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    mask = grad_cam(input, target_index)        # [224, 224]

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), 'gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')
