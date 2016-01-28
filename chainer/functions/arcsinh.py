import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


class Arcsinh(function.Function):

    @property
    def label(self):
        return 'arcsinh'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        y = utils.force_array(numpy.arcsinh(x[0]))
        return y,

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        cuda.elementwise('''
            {ctype}* y, const {ctype}* x
          '''.format(ctype=self.ctype), '''
            y[i] = log(x[i] + sqrt(float(1) + pow(x[i], 2)))
          ''', 'arcsinh_fwd')(y, x[0])
        return y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] / numpy.sqrt(float(1) + x[0]**2)
        if self.cplx:
            cgx = gy[0] / numpy.conj(numpy.sqrt(float(1) + x[0]**2))
        else:
            cgx = None
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = gy[0] / cuda.cumath.sqrt(float(1) + x[0]**2)
        if self.cplx:
            cgx = gy[0] / cuda.cumath.sqrt(float(1) + x[0]**2).conj()
        else:
            cgx = None
        return (gx,), (cgx,)


def arcsinh(x):
    """Elementwise inverse hyperbolic sine function."""
    return Arcsinh()(x)
