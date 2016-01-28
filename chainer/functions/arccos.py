import numpy

from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


class Arccos(function.Function):

    @property
    def label(self):
        return 'arccos'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == numpy.complex64)

    def forward_cpu(self, x):
        y = utils.force_array(numpy.arccos(x[0]))
        return y,

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        cuda.elementwise('''
            {ctype}* y, const {ctype}* x
          '''.format(ctype=self.ctype), '''
            {ctype} j = {ctype}(0, 1);
            y[i] = float(0.5) * float(M_PI) + j * log(j * x[i] + sqrt(float(1) - pow(x[i], 2)))
          '''.format(ctype=self.ctype), 'arccos_fwd')(y, x[0])
        return y,

    def backward_cpu(self, x, gy, cgy):
        gx = -gy[0] / numpy.sqrt(float(1) - x[0]**2)
        cgx = -gy[0] / numpy.conj(numpy.sqrt(float(1) - x[0]**2))
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = -gy[0] / cuda.cumath.sqrt(float(1) - x[0]**2)
        cgx = -gy[0] / cuda.cumath.sqrt(float(1) - x[0]**2).conj()
        return (gx,), (cgx,)


def arccos(x):
    """Elementwise inverse cosine function."""
    return Arccos()(x)
