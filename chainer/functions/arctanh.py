import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Arctanh(function.Function):

    """Inverse hyperbolic tangent function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        y = numpy.arctanh(x[0])
        return y,

    def forward_gpu(self, x):
        y = cuda.empty_like(x[0])
        cuda.elementwise('''
            {ctype}* y, const {ctype}* x
          '''.format(ctype=self.ctype), '''
            y[i] = float(0.5) * (log(float(1) + x[i]) - log(float(1) - x[i]))
          ''', 'arctanh_fwd')(y, x[0])
        return y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] / (1. - x[0]**2)        
        cgx = cgy[0] / numpy.conj(1. - x[0]**2)
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(x[0])
        if self.cplx:
            cgx = cuda.empty_like(x[0])
            cuda.elementwise('''
                {ctype}* gx, {ctype}* cgx, const {ctype}* x, 
                const {ctype}* gy, const {ctype}* cgy
              '''.format(ctype=self.ctype), '''
                gx[i]  = gy[i] / (float(1) - pow(x[i], 2));
                cgx[i] = cgy[i] / conj(float(1) - pow(x[i], 2))
              ''', 'arctanh_bwd')(gx, cgx, x[0], gy[0], cgy[0])
        else:
            cgx = None
            cuda.elementwise('''
                {ctype}* gx, const {ctype}* x, 
                const {ctype}* gy
              '''.format(ctype=self.ctype), '''
                gx[i]  = gy[i] / (float(1) - pow(x[i], 2))
              ''', 'arctanh_bwd')(gx, x[0], gy[0])
        outputs = (gx,), (cgx,)
        return outputs


def arctanh(x):
    """Elementwise inverse hyperbolic tangent function.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Arctanh()(x)
