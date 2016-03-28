import numpy

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import type_check

if cudnn.available:
    from chainer.cudnn import libcudnn
    _mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_SIGMOID']


class Sigmoid(function.Function):

    """Logistic sigmoid function."""

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        self.y = 1 / (1 + numpy.exp(-x[0]))
        return self.y,

    def forward_gpu(self, x):
        if not hasattr(self, 'cplx'):
            self.cplx = x[0].dtype == numpy.complex64
            if self.cplx:
                self.dtype = numpy.complex64
                self.dtype = 'pycuda::complex<float>'
            else:
                self.dtype = numpy.float32
                self.ctype = 'float'

        self.y = cuda.empty_like(x[0])
        if False and cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], 1, 1)
            libcudnn.cudnnActivationForward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(self.y))
        else:
            cuda.elementwise(
                '{ctype}* y, const {ctype}* x'.format(ctype=self.ctype), 
                'y[i] = float(1) / (float(1) + exp(-x[i]))',
                'sigmoid_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] * self.y * (1 - self.y)
        if self.cplx:
            cgx = cgy[0] * numpy.conj(gx)
        else:
            cgx = None
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(x[0])
        if False and cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(self.y, 1, 1)
            libcudnn.cudnnActivationBackward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
                desc.value, cudnn.get_ptr(
                    gy[0]), desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(gx))
        else:
            cuda.elementwise(
                '''
                   {ctype}* gx, const {ctype}* y, 
                   const {ctype}* gy
                '''.format(ctype=self.ctype),
                'gx[i] = gy[i] * y[i] * (float(1) - y[i])',
                'sigmoid_bwd')(gx, self.y, gy[0])
        if self.cplx:
            cgx = cgy[0] * gx.conj()
        else:
            cgx = None
        return (gx,), (cgx,)


def sigmoid(x, use_cudnn=True):
    """Elementwise sigmoid logistic function :math:`f(x)=(1 + \\exp(-x))^{-1}`.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Sigmoid(use_cudnn)(x)
