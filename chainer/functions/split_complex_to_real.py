import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SplitComplexToReal(function.Function):
    """
    Splits an N-dimensional complex-valued layer to a 2N-dimmensional real-valued layer by 
    returning the real and imginary parts.
    """
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size()==1)
        type_check.expect(in_types[0].dtype==numpy.complex64)
        

    def forward_cpu(self, x):
        self.rdim, self.cdim = x[0].shape
        return numpy.hstack([x[0].real, x[0].imag]),


    def forward_gpu(self, x):
        self.rdim, self.cdim = x[0].shape
        y = cuda.empty([self.rdim, self.cdim*2], dtype=numpy.float32)
        cuda.elementwise('float* y, const pycuda::complex<float>* x, int cdim',
                         '''
                            int r = i / (cdim * 2);
                            int c = i % (cdim * 2);
                            int idx = r*cdim + c%cdim;
                            y[i] = c < cdim ? x[idx].real() : x[idx].imag();
                         ''',
                         'split_complex_to_real_fwd')(y, x[0], self.cdim)
        return y,
    

    def backward_cpu(self, x, gy, cgy):
        return ((gy[0][:,:self.cdim]+1j*gy[0][:,self.cdim:],), 
                (gy[0][:,:self.cdim]+1j*gy[0][:,self.cdim:],))
    

    def backward_gpu(self, x, gy, cgy):
        gx = cuda.empty([self.rdim, self.cdim], dtype=numpy.complex64)
        cgx = cuda.empty([self.rdim, self.cdim], dtype=numpy.complex64)
        cuda.elementwise('''pycuda::complex<float>* gx, pycuda::complex<float>* cgx, 
                            const float* gy, int cdim''',
                         '''
                            int r = i / cdim;
                            int c = i % cdim;
                            int idx = r*(cdim*2) + c;
                            gx[i] = pycuda::complex<float>(gy[idx], gy[idx+cdim]);
                            cgx[i] = pycuda::complex<float>(gy[idx], gy[idx+cdim]);
                         ''',
                         'split_complex_to_real_bwd')(gx, cgx, gy[0], self.cdim)
        return (gx,), (cgx,)



def split_complex_to_real(x):
    return SplitComplexToReal()(x)

################################################################################
##########                      USED FOR TESTING                      ##########
################################################################################
# rdim = np.random.randint(1000)
# cdim = np.random.randint(1000)
# data = ((np.random.randn(rdim, cdim) + 
#          1j*np.random.randn(rdim, cdim)).astype(np.complex64))

# arr = Variable(data)
# splitter = F.SplitComplexToReal()
# split = splitter(arr)
# loss = F.mean_squared_error(split, 
#                             Variable(np.float32(np.zeros_like(split.data))))
# loss.backward(retain_grad=True)
# np.array_equal(split.data[:,:cdim] + 1j*split.data[:,cdim:], arr.data)
# np.allclose(arr.data, arr.grad*arr.data.size)
# np.allclose(arr.data, arr.conj_grad*arr.data.size)

# arr = Variable(cuda.to_gpu(data))
# splitter = F.SplitComplexToReal()
# split = splitter(arr)
# loss = F.mean_squared_error(split, 
#                             Variable(cuda.to_gpu(
#                                 np.float32(np.zeros_like(split.data)))))
# loss.backward(retain_grad=True)
# np.array_equal(cuda.to_cpu(split.data)[:,:cdim] + 
#                1j*cuda.to_cpu(split.data)[:,cdim:], cuda.to_cpu(arr.data))
# np.allclose(cuda.to_cpu(arr.data), cuda.to_cpu(arr.grad)*arr.data.size)
# np.allclose(cuda.to_cpu(arr.data), cuda.to_cpu(arr.conj_grad)*arr.data.size)
