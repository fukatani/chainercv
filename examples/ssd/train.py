import argparse
import copy
import numpy as np
import cupy

import chainer
from chainer.datasets import ConcatenatedDataset
from chainer.datasets import TransformDataset
from chainer.optimizer import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainercv.datasets import voc_bbox_label_names
from chainercv.datasets import VOCBboxDataset
from chainercv.datasets import SiameseDataset
from chainercv.extensions import DetectionVOCEvaluator
from chainercv.links.model.ssd import GradientScaling
from chainercv.links.model.ssd import multibox_loss
from chainercv.links import DSSD300
from chainercv.links import ESSD300
from chainercv.links import ESSDResidual300
from chainercv.links import SSD300
from chainercv.links import SSD300Plus
from chainercv.links import SSD512
from chainercv.links import RefineDet320
from chainercv import transforms

from chainercv.links.model.ssd import random_crop_with_bbox_constraints
from chainercv.links.model.ssd import random_distort
from chainercv.links.model.ssd import resize_with_random_interpolation


class MultiboxTrainChain(chainer.Chain):

    def __init__(self, model, alpha=1, k=3):
        super(MultiboxTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.alpha = alpha
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        mb_locs, mb_confs = self.model(imgs)
        loc_loss, conf_loss = multibox_loss(
            mb_locs, mb_confs, gt_mb_locs, gt_mb_labels, self.k)
        loss = loc_loss * self.alpha + conf_loss

        chainer.reporter.report(
            {'loss': loss, 'loss/loc': loc_loss, 'loss/conf': conf_loss},
            self)

        return loss


class RefineDetTrainChain(chainer.Chain):

    def __init__(self, model, k=3):
        super(RefineDetTrainChain, self).__init__()
        with self.init_scope():
            self.model = model
        self.k = k

    def __call__(self, imgs, gt_mb_locs, gt_mb_labels):
        arm_locs, arm_confs, odm_locs, odm_confs = self.model(imgs)

        gt_objectness_label = gt_mb_labels.copy()
        gt_objectness_label[gt_objectness_label > 0] = 1

        arm_loc_loss, arm_conf_loss = multibox_loss(
            arm_locs, arm_confs, gt_mb_locs.copy(), gt_objectness_label, self.k,
            two_class=True)

        objectness = cupy.zeros_like(arm_confs.array)
        objectness[arm_confs.array >= 0.01] = 1
        odm_loc_loss, odm_conf_loss = multibox_loss(
            odm_locs, odm_confs, gt_mb_locs, gt_mb_labels, self.k,
            objectness=objectness)
        loss = arm_loc_loss + arm_conf_loss + odm_loc_loss + odm_conf_loss

        chainer.reporter.report(
            {'loss': loss,
             'arm_loss/loc': arm_loc_loss,
             'arm_loss/conf': arm_conf_loss,
             'odm_loss/loc': odm_loc_loss,
             'odm_loss/conf': odm_conf_loss}, self)

        return loss


class Transform(object):

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

        img, bbox, label = in_data

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
        mb_loc, mb_label = self.coder.encode(bbox, label)

        return img, mb_loc, mb_label


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
        mixup_loc = np.zeros((8732, 4), dtype=np.float32)
        mixup_label = np.zeros((8732, 21), dtype=np.float32)

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
            mixup_loc += mb_loc * weight

            temp_label = np.eye(21)[mb_label]
            temp_label *= weight
            mixup_label += temp_label

        return mixup_img, mixup_loc, mixup_label


class MixupSameTransform(object):

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
        mixup_loc = np.zeros((8732, 4), dtype=np.float32)
        mixup_label = np.zeros((8732, 21), dtype=np.float32)

        data = data0
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
        for weight in (weight0, weight1):
            img -= self.mean
            mixup_img += img * weight

            mb_loc, mb_label = self.coder.encode(bbox, label)
            mixup_loc += mb_loc * weight

            temp_label = np.eye(21)[mb_label]
            temp_label *= weight
            mixup_label += temp_label

        return mixup_img, mixup_loc, mixup_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', choices=('ssd300', 'ssd512', 'ssd300plus', 'dssd300',
                            'essd300', 'essd300residual', 'refinedet320'),
        default='ssd300')
    parser.add_argument('--batchsize', type=int, default=24)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out', default='result')
    parser.add_argument('--mixup', action='store_true', default=False)
    parser.add_argument('--resume')
    args = parser.parse_args()

    if args.model == 'ssd300':
        model = SSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'ssd512':
        model = SSD512(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'ssd300plus':
        model = SSD300Plus(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'dssd300':
        model = DSSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'essd300':
        model = ESSD300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'essd300residual':
        model = ESSDResidual300(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')
    elif args.model == 'refinedet320':
        model = RefineDet320(
            n_fg_class=len(voc_bbox_label_names),
            pretrained_model='imagenet')

    model.use_preset('evaluate')
    if isinstance(model, RefineDet320):
        train_chain = RefineDetTrainChain(model)
    else:
        train_chain = MultiboxTrainChain(model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    if args.mixup:
        base_dataset = ConcatenatedDataset(
            VOCBboxDataset(year='2007', split='trainval'),
            VOCBboxDataset(year='2012', split='trainval')
        )
        train = TransformDataset(
            SiameseDataset(base_dataset, base_dataset),
            MixupTransform(model.coder, model.insize, model.mean))
            # MixupSameTransform(model.coder, model.insize, model.mean))
    else:
        train = TransformDataset(
            ConcatenatedDataset(
                VOCBboxDataset(year='2007', split='trainval'),
                VOCBboxDataset(year='2012', split='trainval')
            ),
            Transform(model.coder, model.insize, model.mean))
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, shared_mem=4000000)
    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    test = VOCBboxDataset(
        year='2007', split='test',
        use_difficult=True, return_difficult=True)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # initial lr is set to 1e-3 by ExponentialShift
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(train_chain)
    for param in train_chain.params():
        if param.name == 'b':
            param.update_rule.add_hook(GradientScaling(2))
        else:
            param.update_rule.add_hook(WeightDecay(0.0005))

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (120000, 'iteration'), args.out)
    trainer.extend(
        extensions.ExponentialShift('lr', 0.1, init=1e-3),
        trigger=triggers.ManualScheduleTrigger([80000, 100000], 'iteration'))

    trainer.extend(
        DetectionVOCEvaluator(
            test_iter, model, use_07_metric=True,
            label_names=voc_bbox_label_names),
        trigger=(10000, 'iteration'))

    log_interval = 10, 'iteration'
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'lr',
         'main/loss', 'main/arm_loss/loc', 'main/arm_loss/conf',
         'main/odm_loss/loc', 'main/odm_loss/conf',
         'validation/main/map']),
        trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(model, 'model_iter_{.updater.iteration}'),
        trigger=(120000, 'iteration'))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
