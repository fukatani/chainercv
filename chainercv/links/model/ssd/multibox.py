import chainer
import chainer.functions as F
from chainer import initializers
import chainer.links as L


class Multibox(chainer.Chain):
    """Multibox head of Single Shot Multibox Detector.

    This is a head part of Single Shot Multibox Detector [#]_.
    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(Multibox, self).__init__()
        with self.init_scope():
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(xs):
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class ResidualMultibox(chainer.Chain):
    """Multibox head of Deconvolutional Single Shot Detector.

    This is a head part of Deconvolutional Single Shot Detector, also usable
    as a head part of Single Shot Multibox Detector[#]_.

    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi,
       Alexander C. Berg
       DSSD : Deconvolutional Single Shot Detector.
       https://arxiv.org/abs/1701.06659.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(ResidualMultibox, self).__init__()
        with self.init_scope():
            self.res = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.res.add_link(Residual, **init)
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(xs):
            x = self.res[i](x)
            mb_loc = self.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class Residual(chainer.Chain):

    """A bottleneck layer that reduces the resolution of the feature map.
    Args:
        initialW (4-D array): Initial weight value used in
            the convolutional layers.
        initial_bias (4-D array): Initial bias value used in
            the convolutional layers.
    """

    def __init__(self, initialW=None, initial_bias=None):
        super(Residual, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                256, 1, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn1 = L.BatchNormalization(256)
            self.conv2 = L.Convolution2D(
                256, 1, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn2 = L.BatchNormalization(256)
            self.conv3 = L.Convolution2D(
                256, 1, pad=1,initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn3 = L.BatchNormalization(1024)
            self.conv4 = L.Convolution2D(
                1024, 1, pad=1, initialW=initialW, initial_bias=initial_bias,
                nobias=True)
            self.bn4 = L.BatchNormalization(1024)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class DeconvolutionalResidualMultibox(chainer.Chain):
    """Multibox head of Deconvolutional Single Shot Detector.

    This is a head part of Deconvolutional Single Shot Detector, also usable
    as a head part of Single Shot Multibox Detector[#]_.

    This link computes :obj:`mb_locs` and :obj:`mb_confs` from feature maps.
    :obj:`mb_locs` contains information of the coordinates of bounding boxes
    and :obj:`mb_confs` contains confidence scores of each classes.

    .. [#] Cheng-Yang Fu, Wei Liu, Ananth Ranga, Ambrish Tyagi,
       Alexander C. Berg
       DSSD : Deconvolutional Single Shot Detector.
       https://arxiv.org/abs/1701.06659.

    Args:
        n_class (int): The number of classes possibly including the background.
        aspect_ratios (iterable of tuple or int): The aspect ratios of
            default bounding boxes for each feature map.
        initialW: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.LeCunUniform`.
        initial_bias: An initializer used in
            :meth:`chainer.links.Convolution2d.__init__`.
            The default value is :class:`chainer.initializers.Zero`.

    """

    def __init__(
            self, n_class, aspect_ratios,
            initialW=None, initial_bias=None):
        self.n_class = n_class
        self.aspect_ratios = aspect_ratios

        super(ResidualMultibox, self).__init__()
        with self.init_scope():
            self.res = chainer.ChainList()
            self.loc = chainer.ChainList()
            self.conf = chainer.ChainList()

        if initialW is None:
            initialW = initializers.LeCunUniform()
        if initial_bias is None:
            initial_bias = initializers.Zero()
        init = {'initialW': initialW, 'initial_bias': initial_bias}

        for ar in aspect_ratios:
            n = (len(ar) + 1) * 2
            self.res.add_link(Residual, **init)
            self.loc.add_link(L.Convolution2D(n * 4, 3, pad=1, **init))
            self.conf.add_link(L.Convolution2D(
                n * self.n_class, 3, pad=1, **init))

    def __call__(self, xs):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        mb_locs = list()
        mb_confs = list()
        for i, x in enumerate(reversed(xs)):
            if i == 0:
                y = x
            else:
                y = self.dec[i](y, x)
            y = self.res[i](y)
            mb_loc = self.loc[i](y)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.conf[i](y)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.n_class))
            mb_confs.append(mb_conf)

        mb_locs = F.concat(mb_locs, axis=1)
        mb_confs = F.concat(mb_confs, axis=1)

        return mb_locs, mb_confs


class DeconvolutionModule(chainer.Chain):
    def __init__(self):
        init = {
            'initialW': initializers.LeCunUniform(),
            'initial_bias': initializers.Zero(),
        }
        super(DeconvolutionModule, self).__init__()
        with self.init_scope():
            self.conv1_1 = L.Convolution2D(512, 1, pad=1, **init)
            self.conv1_2 = L.Convolution2D(512, 3, pad=1, **init)
            self.bn1_1 = L.BatchNormalization(512)
            self.bn1_2 = L.BatchNormalization(512)

            self.deconv2_1 = L.DeConvolution2D(512, 2, **init)
            self.bn2_1 = L.BatchNormalization(512)
            self.conv2_1 = L.Convolution2D(512, 3, pad=1, **init)

    def __call__(self, x1, x2):
        """Compute loc and conf from feature maps

        This method computes :obj:`mb_locs` and :obj:`mb_confs`
        from given feature maps.

        Args:
            xs (iterable of chainer.Variable): An iterable of feature maps.
                The number of feature maps must be same as the number of
                :obj:`aspect_ratios`.

        Returns:
            tuple of chainer.Variable:
            This method returns two :obj:`chainer.Variable`: :obj:`mb_locs` and
            :obj:`mb_confs`.

            * **mb_locs**: A variable of float arrays of shape \
                :math:`(B, K, 4)`, \
                where :math:`B` is the number of samples in the batch and \
                :math:`K` is the number of default bounding boxes.
            * **mb_confs**: A variable of float arrays of shape \
                :math:`(B, K, n\_fg\_class + 1)`.

        """

        x1 = self.conv1_1(x1)
        x1 = self.bn1_1(x1)
        x1 = F.relu(x1)
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)

        x2 = self.deconv2_1(x2)
        x2 = self.conv2_1(x2)
        x2 = self.bn2_1(x2)

        return F.relu(x1 * x2)
