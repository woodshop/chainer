import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Tan(function.Function):

    """Inverse tangent function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        self.y = numpy.tan(x[0])
        return self.y,

    def forward_gpu(self, x):
        self.y = cuda.empty_like(x[0])
        cuda.elementwise('''
            {ctype}* y, const {ctype}* x
          '''.format(ctype=self.ctype), '''
            y[i] = tan(x[i])
          ''', 'tan_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] * (1 + self.y**2)
        if self.cplx:
            cgx = cgy[0] * numpy.conj(1 + self.y**2)
        else:
            cgx = None
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(self.y)
        if self.cplx:
            cgx = cuda.empty_like(self.y)
            cuda.elementwise('''
                {ctype}* gx, {ctype}* cgx, const {ctype}* y, 
                const {ctype}* gy, const {ctype}* cgy
              '''.format(ctype=self.ctype), '''
                gx[i]  = gy[i] * (float(1) + pow(y[i], 2));
                cgx[i] = cgy[i] * conj(float(1) + pow(y[i], 2))
              ''', 'tan_bwd')(gx, cgx, self.y, gy[0], cgy[0])
        else:
            cgx = None
            cuda.elementwise('''
                {ctype}* gx, const {ctype}* y, 
                const {ctype}* gy
              '''.format(ctype=self.ctype), '''
                gx[i]  = gy[i] * (float(1) + pow(y[i], 2))
              ''', 'tan_bwd')(gx, self.y, gy[0])

        outputs = (gx,), (cgx,)
        return outputs


def tan(x):
    """Elementwise tangent function.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Tan()(x)
