#from chainer import cuda, Function, FunctionSet, gradient_check, Variable, \
#    optimizers
from chainer import function
from chainer import cuda
from chainer.utils import type_check
import numpy as np

_preamble = '''\
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


    def __init__(self, gamma=0.5):
        self.gamma = gamma
        super(function.Function, self).__init__()

        
    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == np.complex64,
            in_types[1].dtype == np.complex64,
            in_types[0].shape == in_types[1].shape
        )

        
    def forward_cpu(self, inputs):
        predictions, targets = inputs
        ndim = predictions.shape[1]
        u = predictions[:, :ndim/2]
        v = predictions[:, ndim/2:]
        x_hat = u/(u+v)
        y_hat = v/(u+v)
        x = targets[:, :ndim/2]
        y = targets[:, ndim/2:]
        gamma = np.asarray([self.gamma], dtype=np.complex64)
        def L2(a, b):
            diff = a - b
            diff = diff.ravel()
            return np.array(diff.dot(diff.conj()) / diff.size,
                            np.complex64)
        ret = (L2(x_hat, x) - gamma*L2(x_hat, y) +
               L2(y_hat, y) - gamma*L2(y_hat, x))
        return ret,

    
    def forward_gpu(self, inputs):
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
            'cost_fwd', preamble=_preamble)(ret, x0, x1, ndim, gamma)
        ret = (cuda.gpuarray.sum(ret) / float(x0.size))
        return ret,


    def backward_cpu(self, inputs, gy):
        predictions, targets = inputs
        ndim = predictions.shape[1] / 2
        u = predictions[:, :ndim]
        v = predictions[:, ndim:]
        x_hat = u/(u+v)
        y_hat = v/(u+v)
        x = targets[:, :ndim]
        y = targets[:, ndim:]
        coeff = gy[0] / np.float(x.size)
        #dC_dx_hat = np.conj(2 *
        #                    (((1 - self.gamma) * x_hat + self.gamma * y - x) -
        #                     ((1 - self.gamma) * y_hat + self.gamma * x - y)))
        dC_dx_hat = 2 * (np.conj(x_hat - x) - self.gamma*np.conj(x_hat - y))
        dx_hat_du = y_hat/(u+v)
        #dC_dx = np.conj(2 * ((1 - self.gamma) * x +
        #                       self.gamma * y_hat - x_hat))
        
        #dC_dy_hat = np.conj(2 *
        #                (((1 - self.gamma) * y_hat + self.gamma * x - y) -
        #                 ((1 - self.gamma) * x_hat + self.gamma * y - x)))
        dc_dy_hat = 2 * (np.conj(y_hat - y) - self.gamma*np.conj(y_hat - x))
        dy_hat_dv = x_hat/(u+v)
        #dC_dy = np.conj(2 * (( 1 - self.gamma) * y +
        #                       self.gamma * x_hat - y_hat))
        #return (coeff * np.hstack([dC_dx_hat*dx_hat_du, dC_dy_hat*dy_hat_dv]),
        #        coeff * np.hstack([dC_dx, dC_dy]))
        gx0 = coeff * np.hstack([dC_dx_hat*dx_hat_du, dC_dy_hat*dy_hat_dv])
        return (gx0, -gx0)

    
    def backward_gpu(self, inputs, gy):
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
               //pycuda::complex<float> gamma_i = 
               //    pycuda::complex<float>(1., 0.) - *gamma;
               //gx0[i] = *coeff * conj((gamma_i * x_hat + *gamma * y - x) - 
               //              (gamma_i * y_hat + *gamma * x - y)) *
               //              y_hat / (u + v);
               gx0[i] = *coeff * 
                            (conj(x_hat - x) - *gamma * conj(x_hat - y)) * 
                            y_hat / (u + v);
               //gx1[i] = *coeff * conj(gamma_i * x + *gamma * y_hat - x_hat);
               gx1[i] = -gx0[i];
            ''',
            'cost_bwd', preamble=_preamble)(gx0, gx1, x0, x1, ndim, gamma,
                                            coeff)
        return gx0, gx1
