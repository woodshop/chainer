import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Dropout(function.Function):

    """Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forwrad(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        scale = numpy.float32(1. / (1 - self.dropout_ratio))
        if self.cplx:
            scale = numpy.complex64(scale + 0j)
        self.mask = scale * \
            (numpy.random.rand(*x[0].shape) >= self.dropout_ratio)
        return x[0] * self.mask,

    def forward_gpu(self, x):
        self.rand = cuda.empty(x[0].shape, numpy.float32)
        y = cuda.empty_like(x[0])

        cuda.get_generator().fill_uniform(self.rand)
        self.scale = numpy.float32(1. / (1 - self.dropout_ratio))
        if self.cplx:
            self.scale = numpy.complex64(self.scale + 0j)

        self.kernel = cuda.elementwise(
            '''{ctype}* y, const {ctype}* x, const float* rand, float dropout_ratio,
               {ctype} scale'''.format(ctype=self.ctype),
            'y[i] = rand[i] < dropout_ratio ? {ctype}(0) : scale * x[i]'.format(ctype=self.ctype),
            'dropout')
        self.kernel(y, x[0], self.rand, numpy.float32(self.dropout_ratio), self.scale)
        return y,

    def backward_cpu(self, x, gy, cgy):
        return (gy[0] * self.mask,), (cgy[0] * self.mask,)

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(gy[0])
        cgx = cuda.empty_like(cgy[0])
        self.kernel(gx, gy[0], self.rand, numpy.float32(self.dropout_ratio), self.scale)
        self.kernel(cgx, cgy[0], self.rand, numpy.float32(self.dropout_ratio), self.scale)
        return (gx,), (cgx,)


def dropout(x, ratio=.5, train=True):
    """Drops elements of input variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If True, executes dropout. Otherwise, does nothing.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by G. Hinton: `Improving neural networks by preventing \
    co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.

    """
    if train:
        return Dropout(ratio)(x)
    return x


# class CplxDropout(Dropout):

#     """Dropout regularization."""

#     def check_type_forwrad(self, in_types):
#         type_check.expect(in_types.size() == 1)
#         type_check.expect(in_types[0].dtype == numpy.complex64)

#     def forward_cpu(self, x):
#         scale = numpy.complex64(1. / (1 - self.dropout_ratio))
#         self.mask = scale * \
#             (numpy.random.rand(*x[0].shape) >= self.dropout_ratio)
#         return x[0] * self.mask,

#     def forward_gpu(self, x):
#         self.rand = cuda.empty(x[0].shape, dtype=numpy.float32)
#         y = cuda.empty_like(x[0])

#         cuda.get_generator().fill_uniform(self.rand)
#         self.scale = numpy.complex64(1. / (1 - self.dropout_ratio))

#         self.kernel = cuda.elementwise(
#             '''
#                pycuda::complex<float>* y,
#                const pycuda::complex<float>* x,
#                const float* rand,
#                float dropout_ratio,
#                pycuda::complex<float> scale
#             ''',
#             '''y[i] = rand[i] < dropout_ratio ? 
#                         pycuda::complex<float>(0) : scale * x[i]''',
#             'dropout')
#         self.kernel(y, x[0], self.rand, self.dropout_ratio, self.scale)
#         return y,

#     def backward_cpu(self, x, gy):
#         return gy[0] * self.mask,

#     def backward_gpu(self, x, gy):
#         gx = cuda.empty_like(gy[0])
#         self.kernel(gx, gy[0], self.rand, self.dropout_ratio, self.scale)
#         return gx,

    
# def cplx_dropout(x, ratio=.5, train=True):
#     """Drops elements of input variable randomly.

#     This function drops input elements randomly with probability ``ratio`` and
#     scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
#     mode, it does nothing and just returns ``x``.

#     Args:
#         x (~chainer.Variable): Input variable.
#         ratio (float): Dropout ratio.
#         train (bool): If True, executes dropout. Otherwise, does nothing.

#     Returns:
#         ~chainer.Variable: Output variable.

#     See the paper by G. Hinton: `Improving neural networks by preventing \
#     co-adaptation of feature detectors <http://arxiv.org/abs/1207.0580>`_.

#     """
#     if train:
#         return CplxDropout(ratio)(x)
#     return x
