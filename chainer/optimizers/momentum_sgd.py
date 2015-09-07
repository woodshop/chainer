import numpy

from chainer import cuda
from chainer import optimizer


class MomentumSGD(optimizer.Optimizer):

    """Classical momentum SGD."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def init_state_cpu(self, param, grad):
        return numpy.zeros_like(param)

    def init_state_gpu(self, param, grad):
        return cuda.zeros_like(param)

    def update_one_cpu(self, param, grad, v):
        assert param.dtype == numpy.float32
        assert grad.dtype == numpy.float32
        v *= self.momentum
        v -= self.lr * grad
        param += v

    def update_one_gpu(self, param, grad, v):
        cuda.elementwise(
            '''float* param, const float* grad, float* v,
               float lr, float momentum''',
            '''v[i] = momentum * v[i] - lr * grad[i];
               param[i] += v[i];''',
            'momentum_sgd')(param, grad, v, self.lr, self.momentum)


class CplxMomentumSGD(MomentumSGD):

    def update_one_gpu(self, param, grad, v):
        cuda.elementwise(
            '''
               pycuda::complex<float>* param, 
               const pycuda::complex<float>* grad, 
               pycuda::complex<float>* v,
               float lr, 
               float momentum
            ''',
            '''v[i] = momentum * v[i] - lr * grad[i];
               param[i] += v[i];''',
            'momentum_sgd')(param, grad, v, self.lr, self.momentum)

    def compute_grads_norm(self):
        sqnorm = 0
        for _, g, _ in self.tuples:
            sqnorm += _sqnorm(g)
        return numpy.sqrt(sqnorm)
                    
def _sqnorm(x):
    if isinstance(x, cuda.GPUArray):
        with cuda.using_device(x):
            return numpy.real(cuda.gpuarray.dot(x, x.conj()).get())
    x = x.ravel()
    return float(x.dot(x.conj()))
                                    
