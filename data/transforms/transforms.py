# encoding: utf-8

import math
import random


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability     # 0.5
        self.mean = mean
        self.sl = sl       # 0.02
        self.sh = sh       # 0.4
        self.r1 = r1       # 0.3

    def __call__(self, img):       # [3, 256, 128]

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]      # 256 * 128 = 32768

            target_area = random.uniform(self.sl, self.sh) * area      # 6675.79
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)        # 2.1

            h = int(round(math.sqrt(target_area * aspect_ratio)))      # 118
            w = int(round(math.sqrt(target_area / aspect_ratio)))      # 56

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img






class RandomSwap(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability     # 0.5
        self.mean = mean
        self.sl = sl       # 0.02
        self.sh = sh       # 0.4
        self.r1 = r1       # 0.3

    def __call__(self, img, swap):       # [3, 256, 128], [3, 256, 128]

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]      # 256 * 128 = 32768

            target_area = random.uniform(self.sl, self.sh) * area      # 6675.79
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)        # 2.1

            h = int(round(math.sqrt(target_area * aspect_ratio)))      # 118
            w = int(round(math.sqrt(target_area / aspect_ratio)))      # 56

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = swap[0, x1:x1 + h, y1:y1 + w]
                    img[1, x1:x1 + h, y1:y1 + w] = swap[1, x1:x1 + h, y1:y1 + w]
                    img[2, x1:x1 + h, y1:y1 + w] = swap[2, x1:x1 + h, y1:y1 + w]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = swap[0, x1:x1 + h, y1:y1 + w]
                return img

        return img






