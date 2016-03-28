import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class Real(function.Function):
    """
    Return real part of complex array.
    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size()==1)
        type_check.expect(in_types[0].dtype==numpy.complex64)
        

    def forward_cpu(self, x):
        return x[0].real,


    def forward_gpu(self, x):
        return x[0].real,
    

    def backward_cpu(self, x, gy, cgy):
        return (gy[0]*(1+0j),), (gy[0]*(1+0j),)
    

    def backward_gpu(self, x, gy, cgy):
        return (gy[0]*(1+0j),), (gy[0]*(1+0j),)



def real(x):
    return Real()(x)

