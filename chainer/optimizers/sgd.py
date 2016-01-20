from chainer import cuda
from chainer import optimizer
import numpy as np

class SGD(optimizer.Optimizer):

    """Vanilla Stochastic Gradient Descent."""

    def __init__(self, lr=0.01, cplx=False):
        self.cplx = cplx
        if cplx:
            self.dtype = np.complex64
            self.ctype = 'pycuda::complex<float>'
        else:
            self.dtype = np.float32
            self.ctype = 'float'
        self.lr = self.dtype(lr)


    def update_one_cpu(self, param, grad, _):
        if self.cplx:
            param -= self.lr * np.conj(grad)
        else:
            param -= self.lr * grad

    def update_one_gpu(self, param, grad, _):
        # ptmp = param.copy()
        # gtmp = grad.copy()
        assert param.dtype == self.dtype
        assert grad.dtype == self.dtype
        if self.cplx:
            cuda.elementwise('''{ctype}* param, const {ctype}* grad, 
                                   {ctype} lr'''.format(ctype=self.ctype),
                             'param[i] -= lr * conj(grad[i])',
                             'sgd')(param, grad, self.lr)
        else:
            cuda.elementwise('''{ctype}* param, const {ctype}* grad, 
                                   float lr'''.format(ctype=self.ctype),
                             'param[i] -= lr * grad[i]',
                             'sgd')(param, grad, self.lr)

        # t = np.allclose(cuda.to_cpu(ptmp) - self.lr * np.conj(cuda.to_cpu(gtmp)),
        #                 cuda.to_cpu(param))
        # if not t:
        #     err = np.max(numpy.abs((cuda.to_cpu(ptmp) - 
        #                   self.lr * np.conj(cuda.to_cpu(gtmp))) - 
        #                  cuda.to_cpu(param)))
        #     print("\tWARNING in sgd: max abs error: {}".format(err)) 
        #     import pdb; pdb.set_trace()
