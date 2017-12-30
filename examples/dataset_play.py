import numpy

from chainercv.links.model.ssd import MultiboxCoderSoftlabel
from chainercv.links.model.ssd import MultiboxCoder


coder = MultiboxCoder(grids=(38, 19, 10, 5, 3, 1),
                      aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
                      steps=(8, 16, 32, 64, 100, 300),
                      sizes=(30, 60, 111, 162, 213, 264, 315),
                      variance=(0.1, 0.2))

image = numpy.zeros((20, 14, 300, 300), dtype=numpy.float32)
bbox = numpy.array([[0, 0, 10, 10]], dtype=numpy.float32)
label = numpy.array([2], dtype=numpy.int32)

mb_loc, mb_label = coder.encode(bbox, label)
pass


# MultiboxCoderSoftlabel(grids=(38, 19, 10, 5, 3, 1),
#                        aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
#                        steps=(8, 16, 32, 64, 100, 300),
#                        sizes=(30, 60, 111, 162, 213, 264, 315),
#                        variance=(0.1, 0.2))
#
# image = numpy.zeros((20, 14, 300, 300))
# labels = numpy.array([2, 3, 4])
# weights = numpy.array([0.3, 0.3, 0.7])
