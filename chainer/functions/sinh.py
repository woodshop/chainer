import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


class Sinh(function.Function):

    @property
    def label(self):
        return 'sinh'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.char.lower() == 'f')

    def forward_cpu(self, x):
        self.y = utils.force_array(numpy.sinh(x[0]))
        return self.y,

    def forward_gpu(self, x):
        y = cuda.cumath.sinh(x[0])
        return y,

    def backward_cpu(self, x, gy, cgy):
        gx = utils.force_array(numpy.cosh(x[0]) * gy[0])
        if self.cplx:
            cgx = utils.force_array(numpy.conj(numpy.cosh(x[0])) * cgy[0])
        else:
            cgx = None
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = utils.force_array(cuda.cumath.cosh(x[0]) * gy[0])
        if self.cplx:
            cgx = utils.force_array(cuda.cumath.cosh(x[0]).conj() * cgy[0])
        else:
            cgx = None
        return (gx,), (cgx,)


def sinh(x):
    """Elementwise hyperbolic sin function."""
    return Sinh()(x)
