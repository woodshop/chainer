from chainer import cuda
from chainer import optimizer
import numpy as np

class SGD(optimizer.Optimizer):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one_cpu(self, param, grad, _):
        param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        assert param.dtype == np.float32
        assert grad.dtype == np.float32
        cuda.elementwise('float* param, const float* grad, float lr',
                         'param[i] -= lr * grad[i]',
                         'sgd')(param, grad, self.lr)

class CplxSGD(SGD):

    def update_one_gpu(self, param, grad, _):
        cuda.elementwise(
            '''
               pycuda::complex<float>* param, 
               const pycuda::complex<float>* grad,
               float lr
            ''',
            'param[i] -= lr * grad[i]',
            'sgd')(param, grad, self.lr)
