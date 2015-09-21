#from chainer import cuda, Function, FunctionSet, gradient_check, Variable, \
#    optimizers
from chainer import function
from chainer import cuda
from chainer.utils import type_check
import numpy as np

    
_preamble_real = '''\
#define COMMON_ROUTINE \
int r = i / ndim; \
int c = i % ndim; \
int j = r * ndim + ((c < ndim/2) ? (c + ndim/2) : (c - ndim/2)); \
float u = x0[i]; \
float v = x0[j]; \
float x = x1[i]; \
float y = x1[j]; \
float x_hat = x0[i] / (x0[i] + x0[j]); \
float y_hat = x0[j] / (x0[i] + x0[j])
'''

_preamble_cplx = '''\
#define COMMON_ROUTINE \
int r = i / ndim; \
int c = i % ndim; \
int j = r * ndim + ((c < ndim/2) ? (c + ndim/2) : (c - ndim/2)); \
pycuda::complex<float> u = x0[i]; \
pycuda::complex<float> v = x0[j]; \
pycuda::complex<float> x = x1[i]; \
pycuda::complex<float> y = x1[j]; \
pycuda::complex<float> x_hat = x0[i] / (x0[i] + x0[j]); \
pycuda::complex<float> y_hat = x0[j] / (x0[i] + x0[j])
'''

class JointMSEMaskingCost(function.Function):


    def __init__(self, gamma=0.5, is_real=True):
        self.gamma = gamma
        self.is_real = is_real
        super(function.Function, self).__init__()
        if is_real:
            self.check_type_forward = self._check_type_forward_real
            #self.forward_cpu =        self._forward_cpu_real
            self.forward_gpu =        self._forward_gpu_real
            #self.backward_cpu =       self._backward_cpu_real
            self.backward_gpu =       self._backward_gpu_real
        else:
            self.check_type_forward = self._check_type_forward_cplx
            #self.forward_cpu =        self._forward_cpu_cplx
            self.forward_gpu =        self._forward_gpu_cplx
            #self.backward_cpu =       self._backward_cpu_cplx
            self.backward_gpu =       self._backward_gpu_cplx

            
    def _check_type_forward_real(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
        )


    def _check_type_forward_cplx(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.complex64,
            in_types[1].dtype == np.complex64,
            in_types[0].shape == in_types[1].shape
        )

        
    def forward_cpu(self, inputs):
        if self.is_real:
            dtype = np.float32
        else:
            dtype = np.complex64
        predictions, targets = inputs
        ndim = predictions.shape[1]
        u = predictions[:, :ndim/2]
        v = predictions[:, ndim/2:]
        x_hat = u/(u+v)
        y_hat = v/(u+v)
        x = targets[:, :ndim/2]
        y = targets[:, ndim/2:]
        gamma = np.asarray([self.gamma], dtype=dtype)
        def L2(a, b):
            diff = a - b
            diff = diff.ravel()
            return np.array(diff.dot(diff.conj()) / diff.size, dtype)
        ret = (L2(x_hat, x) - gamma*L2(x_hat, y) +
               L2(y_hat, y) - gamma*L2(y_hat, x)).squeeze()
        return ret,


    def _forward_gpu_real(self, inputs):
        x0, x1 = inputs
        ndim = x0.shape[1]
        gamma = cuda.to_gpu(np.asarray(self.gamma, dtype=np.float32))
        ret = cuda.empty_like(x0)
        cuda.elementwise(
            '''
               float* ret,
               const float* x0, 
               const float* x1,
               const int ndim,
               const float* gamma
            ''',
            '''
               COMMON_ROUTINE;
               ret[i] = (x_hat - x) * (x_hat - x) - 
                   *gamma * ((x_hat - y) * (x_hat - y)) + 
                   (y_hat - y) * (y_hat - y) - 
                   *gamma * ((y_hat - x) * (y_hat - x))
            ''',
            'cost_fwd', preamble=_preamble_real)(ret, x0, x1, ndim, gamma)
        ret = (cuda.gpuarray.sum(ret) / np.float32(x0.size))
        return ret,


    def _forward_gpu_cplx(self, inputs):
        x0, x1 = inputs
        ndim = x0.shape[1]
        gamma = cuda.to_gpu(np.asarray(self.gamma + 0j, dtype=np.complex64))
        ret = cuda.empty_like(x0)
        cuda.elementwise(
            '''
               pycuda::complex<float>* ret,
               const pycuda::complex<float>* x0, 
               const pycuda::complex<float>* x1,
               const int ndim,
               const pycuda::complex<float>* gamma
            ''',
            '''
               COMMON_ROUTINE;
               ret[i] = (x_hat - x) * conj(x_hat - x) - 
                   *gamma * ((x_hat - y) * conj(x_hat - y)) + 
                   (y_hat - y) * conj(y_hat - y) - 
                   *gamma * ((y_hat - x) * conj(y_hat - x))
            ''',
            'cost_fwd', preamble=_preamble_cplx)(ret, x0, x1, ndim, gamma)
        ret = (cuda.gpuarray.sum(ret) / np.complex64(x0.size))
        return ret,


    def backward_cpu(self, inputs, gy):
        predictions, targets = inputs
        if self.is_real:
            dtype = np.float32
        else:
            dtype = np.complex64
        ndim = predictions.shape[1] / 2
        u = predictions[:, :ndim]
        v = predictions[:, ndim:]
        x_hat = u/(u+v)
        y_hat = v/(u+v)
        x = targets[:, :ndim]
        y = targets[:, ndim:]
        coeff = gy[0] / dtype(x.size)

        dC_dx_hat = 2 * ((np.conj(x_hat - x) - self.gamma*np.conj(x_hat - y)))
        dC_dy_hat = 2 * ((np.conj(y_hat - y) - self.gamma*np.conj(y_hat - x)))
        dx_hat_du = y_hat/(u+v)
        dy_hat_dv = x_hat/(u+v)

        dC_du = (dC_dx_hat - dC_dy_hat) * dx_hat_du
        dC_dv = (dC_dy_hat - dC_dx_hat) * dy_hat_dv
        gx0 = coeff * np.hstack([dC_du, dC_dv])
        return (gx0, -gx0)

    
    def _backward_gpu_real(self, inputs, gy):
        x0, x1 = inputs
        gx0 = cuda.empty_like(x0)
        gx1 = cuda.empty_like(x1)
        ndim = x0.shape[1]
        gamma = cuda.to_gpu(np.asarray(self.gamma + 0j, dtype=np.complex64))
        coeff = 2. * gy[0] / np.float32(x0[:,:ndim/2].size)
        cuda.elementwise(
            '''
            float* gx0, 
            float* gx1, 
            const float* x0, 
            const float* x1,
            const int ndim,
            const float* gamma,
            const float* coeff
            ''',
            '''
               COMMON_ROUTINE;
               gx0[i] = *coeff * ((x_hat - x) - *gamma * (x_hat - y) -
                   (y_hat - y) - *gamma * (y_hat - x) * 
                   y_hat / (u + v));
               gx1[i] = -gx0[i];
            ''',
            'cost_bwd', preamble=_preamble_real)(gx0, gx1, x0, x1, ndim, gamma,
                                                 coeff)
        return gx0, gx1


    def _backward_gpu_cplx(self, inputs, gy):
        x0, x1 = inputs
        gx0 = cuda.empty_like(x0)
        gx1 = cuda.empty_like(x1)
        ndim = x0.shape[1]
        gamma = cuda.to_gpu(np.asarray(self.gamma + 0j, dtype=np.complex64))
        coeff = 2. * gy[0] / float(x0[:,:ndim/2].size)
        cuda.elementwise(
            '''
            pycuda::complex<float>* gx0, 
            pycuda::complex<float>* gx1, 
            const pycuda::complex<float>* x0, 
            const pycuda::complex<float>* x1,
            const int ndim,
            const pycuda::complex<float>* gamma,
            const pycuda::complex<float>* coeff
            ''',
            '''
               COMMON_ROUTINE;
               gx0[i] = *coeff * (conj(x_hat - x) - *gamma * conj(x_hat - y) -
                   conj(y_hat - y) - *gamma * conj(y_hat - x) * 
                   y_hat / (u + v));
               gx1[i] = -gx0[i];
            ''',
            'cost_bwd', preamble=_preamble_cplx)(gx0, gx1, x0, x1, ndim, gamma,
                                                 coeff)
        return gx0, gx1
