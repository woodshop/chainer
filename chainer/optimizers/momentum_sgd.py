import numpy

from chainer import cuda
from chainer import optimizer


class MomentumSGD(optimizer.Optimizer):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9, cplx=False):
        self.lr = lr
        self.momentum = momentum
        self.cplx = cplx
        if cplx:
            self.dtype = numpy.complex64
            self.ctype = 'pycuda::complex<float>'
        else:
            self.dtype = numpy.float32
            self.ctype = 'float'

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, v):
        assert param.dtype == self.dtype
        assert grad.dtype == self.dtype
        v *= self.momentum
        v -= self.lr * grad
        param += v

    def update_one_gpu(self, param, grad, v):
        if self.cplx:
            cuda.elementwise(
                '''{ctype}* param, const {ctype}* grad, {ctype}* v,
                   float lr, float momentum'''.format(ctype=self.ctype),
                '''v[i] = momentum * v[i] - lr * conj(grad[i]);
                   param[i] += v[i];''',
                'momentum_sgd')(param, grad, v, self.lr, self.momentum)
        else:
            cuda.elementwise(
                '''{ctype}* param, const {ctype}* grad, {ctype}* v,
                   float lr, float momentum'''.format(ctype=self.ctype),
                '''v[i] = momentum * v[i] - lr * grad[i];
                   param[i] += v[i];''',
                'momentum_sgd')(param, grad, v, self.lr, self.momentum)
