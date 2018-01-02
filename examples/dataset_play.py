import copy

import numpy as np

from chainer.datasets import TransformDataset
from chainercv.links.model.ssd import MultiboxCoderSoftlabel
from chainercv.links.model.ssd import MultiboxCoder
from chainercv.datasets import SiameseDataset, VOCBboxDataset

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv import transforms


class MixupTransform(object):

    def __init__(self, coder, size, mean):
        # to send cpu, make a copy
        self.coder = copy.copy(coder)
        self.coder.to_cpu()

        self.size = size
        self.mean = mean

    def __call__(self, in_data):
        # There are five data augmentation steps
        # 1. Color augmentation
        # 2. Random expansion
        # 3. Random cropping
        # 4. Resizing with random interpolation
        # 5. Random horizontal flipping

        data0, data1 = in_data

        weight0 = np.random.uniform()
        weight1 = 1.0 - weight0

        mixup_img = np.zeros((data0[0].shape[0], self.size, self.size), dtype=np.float32)
        mixup_locs = np.zeros((8732, 4), dtype=np.float32)
        mixup_labels = np.zeros((8732, 21), dtype=np.float32)

        for data, weight in zip((data0, data1), (weight0, weight1)):
            img, bbox, label = data

            # 1. Color augmentation
            img = random_distort(img)

            # 2. Random expansion
            if np.random.randint(2):
                img, param = transforms.random_expand(
                    img, fill=self.mean, return_param=True)
                bbox = transforms.translate_bbox(
                    bbox, y_offset=param['y_offset'], x_offset=param['x_offset'])

            # 3. Random cropping
            img, param = random_crop_with_bbox_constraints(
                img, bbox, return_param=True)
            bbox, param = transforms.crop_bbox(
                bbox, y_slice=param['y_slice'], x_slice=param['x_slice'],
                allow_outside_center=False, return_param=True)
            label = label[param['index']]

            # 4. Resizing with random interpolatation
            _, H, W = img.shape
            img = resize_with_random_interpolation(img, (self.size, self.size))
            bbox = transforms.resize_bbox(bbox, (H, W), (self.size, self.size))

            # 5. Random horizontal flipping
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (self.size, self.size), x_flip=params['x_flip'])

            # Preparation for SSD network
            img -= self.mean
            mixup_img += img * weight

            mb_loc, mb_label = self.coder.encode(bbox, label)
            mixup_locs += mb_loc * weight

            temp_labels = np.zeros((8732, 21), dtype=np.float32)
            for i in range(20):
                idx = np.where(mb_label == i)
                temp_labels[idx, i] = 1
            temp_labels *= weight
            mixup_labels += temp_labels

        return mixup_img, mixup_locs, mixup_labels

model = SSD300(n_fg_class=len(voc_bbox_label_names),
               pretrained_model='imagenet')
a = VOCBboxDataset(year='2007', split='trainval')

train = TransformDataset(
            SiameseDataset(
            a, a), MixupTransform(model.coder, model.insize, model.mean))

train.get_example(0)

# coder = MultiboxCoder(grids=(38, 19, 10, 5, 3, 1),
#                       aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
#                       steps=(8, 16, 32, 64, 100, 300),
#                       sizes=(30, 60, 111, 162, 213, 264, 315),
#                       variance=(0.1, 0.2))
#
# image = numpy.zeros((20, 14, 300, 300), dtype=numpy.float32)
# bbox = numpy.array([[0, 0, 10, 10]], dtype=numpy.float32)
# label = numpy.array([2], dtype=numpy.int32)
#
# mb_loc, mb_label = coder.encode(bbox, label)


# MultiboxCoderSoftlabel(grids=(38, 19, 10, 5, 3, 1),
#                        aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
#                        steps=(8, 16, 32, 64, 100, 300),
#                        sizes=(30, 60, 111, 162, 213, 264, 315),
#                        variance=(0.1, 0.2))
#
# image = numpy.zeros((20, 14, 300, 300))
# labels = numpy.array([2, 3, 4])
# weights = numpy.array([0.3, 0.3, 0.7])
