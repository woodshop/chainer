import numpy

from chainer import cuda
from chainer import cudnn
from chainer import function
from chainer.utils import type_check

if cudnn.available:
    from chainer.cudnn import libcudnn
    _mode = libcudnn.cudnnActivationMode['CUDNN_ACTIVATION_TANH']


class Tanh(function.Function):

    """Hyperbolic tangent function."""

    def __init__(self, use_cudnn=True, cplx=False):
        self.use_cudnn = use_cudnn
        self.cplx = cplx
        self.dtype = numpy.complex64 if cplx else numpy.float32

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0])
        return self.y,

    def forward_gpu(self, x):
        if self.cplx:
            ctype = 'pycuda::complex<float>'
            tanhf = 'pycuda::tanh'
        else:
            ctype = 'float'
            tanhf = 'tanhf'
        self.y = cuda.empty_like(x[0])
        #import pdb; pdb.set_trace()
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(x[0], 1, 1)
            libcudnn.cudnnActivationForward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(self.y))
        else:
            cuda.elementwise('{ctype}* y, const {ctype}* x'.format(ctype=ctype), 
                             'y[i] = {tanhf}(x[i])'.format(tanhf=tanhf),
                             'tanh_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] * (1 - self.y * self.y)
        cgx = cgy[0] * numpy.conj(1 - self.y * self.y)
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy):
        gx = cuda.empty_like(self.y)
        if cudnn.enabled and self.use_cudnn:
            handle = cudnn.get_default_handle()
            desc = cudnn.get_tensor_desc(self.y, 1, 1)
            libcudnn.cudnnActivationBackward(
                handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
                desc.value, cudnn.get_ptr(
                    gy[0]), desc.value, cudnn.get_ptr(x[0]),
                0, desc.value, cudnn.get_ptr(gx))
        else:
            cuda.elementwise(
                'float* gx, const float* y, const float* gy',
                'gx[i] = gy[i] * (1 - y[i] * y[i])',
                'tanh_bwd')(gx, self.y, gy[0])
        return gx,


def tanh(x, use_cudnn=True, cplx=False):
    """Elementwise hyperbolic tangent function.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Tanh(use_cudnn, cplx=cplx)(x)


# class CplxTanh(Tanh):

#     """Hyperbolic tangent function."""

#     def check_type_forward(self, in_types):
#         type_check.expect(in_types.size() == 1)
#         type_check.expect(in_types[0].dtype == numpy.complex64)

#     def forward_cpu(self, x):
#         self.y = numpy.tanh(x[0])
#         return self.y,

#     def forward_gpu(self, x):
#         self.y = cuda.empty_like(x[0])
#         if cudnn.enabled and self.use_cudnn:
#             handle = cudnn.get_default_handle()
#             desc = cudnn.get_tensor_desc(x[0], 1, 1)
#             libcudnn.cudnnActivationForward(
#                 handle, _mode, 1, desc.value, cudnn.get_ptr(x[0]),
#                 0, desc.value, cudnn.get_ptr(self.y))
#         else:
#             cuda.elementwise(
#                 '''
#                 pycuda::complex<float>* y, 
#                 const pycuda::complex<float>* x
#                 ''',
#                 'y[i] = tanh(x[i])',
#                 'tanh_fwd')(self.y, x[0])
#         return self.y,

#     def backward_cpu(self, x, gy):
#         return gy[0] * (1 - self.y * self.y),

#     def backward_gpu(self, x, gy):
#         gx = cuda.empty_like(self.y)
#         if cudnn.enabled and self.use_cudnn:
#             handle = cudnn.get_default_handle()
#             desc = cudnn.get_tensor_desc(self.y, 1, 1)
#             libcudnn.cudnnActivationBackward(
#                 handle, _mode, 1, desc.value, cudnn.get_ptr(self.y),
#                 desc.value, cudnn.get_ptr(
#                     gy[0]), desc.value, cudnn.get_ptr(x[0]),
#                 0, desc.value, cudnn.get_ptr(gx))
#         else:
#             cuda.elementwise(
#                 '''
#                    pycuda::complex<float>* gx,
#                    const pycuda::complex<float>* y,
#                    const pycuda::complex<float>* gy
#                 ''',
#                 '''
#                    gx[i] = gy[i] * (pycuda::complex<float>(1) - y[i] * y[i]);
#                 ''',
#                 'tanh_bwd')(gx, self.y, gy[0])
#         return gx,


# def cplx_tanh(x, use_cudnn=True):
#     """Elementwise hyperbolic tangent function.

#     Args:
#         x (~chainer.Variable): Input variable.
#         use_cudnn (bool): If True and CuDNN is enabled, then this function uses
#             CuDNN as the core implementation.

#     Returns:
#         ~chainer.Variable: Output variable.

#     """
#     return CplxTanh(use_cudnn)(x)

