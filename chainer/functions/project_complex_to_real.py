import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class ProjectComplexToReal(function.Function):
    """
    Projects a complex-valued layer to a real-valued layer by computing the 
    squared magnitudes.
    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size()==1)
        type_check.expect(in_types[0].dtype==numpy.complex64)
        
    def forward_cpu(self, x):
        return (x[0] * x[0].conj()).real,

    # This code is probably /should be broken. Need to investigate later.
    def forward_gpu(self, x):
        y = cuda.empty(x[0].shape, dtype=numpy.float32)
        cuda.elementwise('float* y, const pycuda::complex<float>* x',
                         'y[i] = (x[i] * conj(x[i])).real()',
                         'project_complex_to_real_fwd')(y, x[0])
        return y,
    
    def backward_cpu(self, x, gy, cgy):
        gx = gy[0] * x[0].conj()
        cgx = gy[0] * x[0]
        return (gx,), (cgx,)
    
    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty(gy[0].shape, dtype=numpy.complex64)
        cgx = cuda.empty(gy[0].shape, dtype=numpy.complex64)
        cuda.elementwise('''pycuda::complex<float>* gx, pycuda::complex<float>* cgx, 
                            const pycuda::complex<float>* x, const float* gy''',
                         '''gx[i] = gy[i] * conj(x[i]);
                            cgx[i] = gy[i] * x[i]''',
                         'project_complex_to_real_bwd')(gx, cgx, x[0], gy[0])
        return (gx,), (cgx,)


def project_complex_to_real(x):
    return ProjectComplexToReal()(x)
