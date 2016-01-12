import numpy

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import type_check


class Georgiou(function.Function):

    """TODO: write doc."""

    def __init__(self, c=1., r=1.):
        self.c = float(c)
        self.r = float(r)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        raise NotImplementedError

    def forward_gpu(self, x):
        self.y = cuda.empty_like(x[0])
        cuda.elementwise(
            '''
               {ctype}* y, const {ctype}* x, const float c, const float r
            '''.format(ctype=self.ctype),
            '''
               y[i] = x[i] / (c + (float)(1.)/r * abs(x[i]))
            '''.format(ctype=self.ctype), 'georgiou_fwd')(self.y, x[0], self.c, 
                                                          self.r)
        return self.y,

    def backward_cpu(self, x, gy, cgy):
        raise NotImplementedError

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(gy[0])
        if self.cplx:
            cgx = cuda.empty_like(cgy[0])
            cuda.elementwise(
                '''
                   {ctype}* gx, {ctype}* cgx, const {ctype}* x, 
                   const float c, const float r
                '''.format(ctype=self.ctype),
                '''
                   gx[i] = r*((float)(2.) * c * r + abs(x[i])) / 
                      (2. * (c * r + abs(x[i])) * (c * r + abs(x[i])));
                   cgx[i] = -r * x[i] * x[i] / 
                              ((float)(2.) * abs(x[i]) * (c * r + abs(x[i])) * 
                                 (c * r + abs(x[i])))
                ''', 'georgiou_bwd')(gx, cgx, x[0], self.c, self.r)
            outputs = ((gy[0]*gx + cgy[0]*cgx.conj(),), 
                       (gy[0]*cgx + cgy[0]*gx.conj(),))
        else:
            cuda.elementwise(
                '''
                   {ctype}* gx, const {ctype}* x, const float c, const float r
                '''.format(ctype=self.ctype),
                '''
                   gx[i] = (c*r*r) / ((c*r + abs(x[i])) * (c*r + abs(x[i])))
                ''', 'giorgiou_bwd')(gx, x[0], self.c, self.r)
            outputs = (gx,), (None,)

        return outputs


def georgiou(x, c=1., r=1.):
    """
       TODO: Doc
    """
    return Georgiou(c, r)(x)
