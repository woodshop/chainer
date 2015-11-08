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

    def __init__(self, use_cudnn=True):
        self.use_cudnn = use_cudnn

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype == self.dtype)

    def forward_cpu(self, x):
        self.y = numpy.tanh(x[0])
        return self.y,

    def forward_gpu(self, x):
        if self.cplx:
            tanhf = 'pycuda::tanh'
        else:
            tanhf = 'tanhf'
        self.y = cuda.empty_like(x[0])
        cuda.elementwise('{ctype}* y, const {ctype}* x'.format(
            ctype=self.ctype), 
                         'y[i] = {tanhf}(x[i])'.format(tanhf=tanhf),
                         'tanh_fwd')(self.y, x[0])
        return self.y,

    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] * (1 - self.y * self.y)
        cgx = cgy[0] * numpy.conj(1 - self.y * self.y)
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty_like(self.y)
        if self.cplx:
            cgx = cuda.empty_like(self.y)
            cuda.elementwise(
                '''{ctype}* gx, {ctype}* cgx, const {ctype}* y, 
                   const {ctype}* gy, const {ctype}* cgy'''.format(
                    ctype=self.ctype),
                '''gx[i]  = gy[i]  * ({ctype}(1.) - y[i] * y[i]);
                   cgx[i] = cgy[i] * conj({ctype}(1.) - y[i] * y[i])'''.format(
                       ctype=self.ctype),
                'tanh_bwd')(gx, cgx, self.y, gy[0], cgy[0])
        else:
            cuda.elementwise(
                '''{ctype}* gx, const {ctype}* y, 
                   const {ctype}* gy'''.format(
                    ctype=self.ctype),
                '''gx[i]  = gy[i]  * (1 - y[i] * y[i])''',
                'tanh_bwd')(gx, self.y, gy[0])
            cgx = None

        outputs = (gx,), (cgx,)

        ### NOT SURE ABOUT THIS (HIGH NUMERICAL ERROR?)
        # tmp = self.y.copy()
        # self.y = cuda.to_cpu(self.y)
        # args = [[cuda.to_cpu(i) for i in inputs] for inputs in [x, gy, cgy]]
        # results = self.backward(*args)
        # self.y = tmp
        # t = [numpy.allclose(r, cuda.to_cpu(o), equal_nan=True) 
        #      for result,output in zip(results, outputs) 
        #      for r,o in zip(*[result, output])]
        # if not all(t):
        #     err = numpy.max([numpy.abs(r - cuda.to_cpu(o)) 
        #            for result,output in zip(results, outputs) 
        #            for r,o in zip(*[result, output])])
        #     print("\tWARNING in tanh: max abs error: {}".format(err)) 
        #     # import pdb; pdb.set_trace()
        return outputs


def tanh(x, use_cudnn=True):
    """Elementwise hyperbolic tangent function.

    Args:
        x (~chainer.Variable): Input variable.
        use_cudnn (bool): If True and CuDNN is enabled, then this function uses
            CuDNN as the core implementation.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Tanh(use_cudnn)(x)
