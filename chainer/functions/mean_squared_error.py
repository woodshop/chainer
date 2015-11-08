import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class MeanSquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    # def __init__(self):
    #     self.cplx = cplx
    #     if cplx:
    #         self.dtype = numpy.complex64
    #         self.ctype = 'pycuda::complex<float>'
    #     else:
    #         self.dtype = numpy.float32
    #         self.ctype = 'float'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == self.dtype,
            in_types[1].dtype == self.dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return numpy.array(diff.dot(numpy.conj(diff)) / diff.size, self.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        if self.cplx:
            ret = cuda.reduce(
                'const {ctype}* x0, const {ctype}* x1'.format(ctype=self.ctype),
                '(x0[i] - x1[i]) * conj((x0[i] - x1[i]))',
                'a+b', '0', 'mse_fwd', self.dtype)(x0, x1)
        else:
            ret = cuda.reduce(
                'const {ctype}* x0, const {ctype}* x1'.format(ctype=self.ctype),
                '(x0[i] - x1[i]) * (x0[i] - x1[i])',
                'a+b', '0', 'mse_fwd', self.dtype)(x0, x1)
        ret /= x0.size
        return ret,

    def backward_cpu(self, inputs, gy, cgy):
        coeff = 2. / self.diff.size
        gx = gy[0] * coeff * self.diff.conj()
        cgx = cgy[0] * coeff * self.diff
        return (gx, -gx), (cgx, -cgx)

    def backward_gpu(self, x, gy, cgy):
        x0, x1 = x
        gx  = cuda.empty_like(x0) 
        coeff = cuda.to_gpu(numpy.asarray(2. / x0.size).astype(self.dtype))
        if self.cplx:
            cgx = cuda.empty_like(x0)
            cuda.elementwise(
                '''{ctype}* gx, {ctype}* cgx, const {ctype}* x0, 
                   const {ctype}* x1, const {ctype}* gy, const {ctype}* cgy, 
                   const {ctype}* coeff'''.format(ctype=self.ctype),
                '''gx[i]  = *gy  * (*coeff) * conj(x0[i] - x1[i]);
                   cgx[i] = *cgy * (*coeff) * (x0[i] - x1[i])''',
                'mse_bwd')(gx, cgx, x0, x1, gy[0], cgy[0], coeff)
            outputs = (gx, -gx), (cgx, -cgx)

            ### THIS CHECKS OUT
            # self.diff = cuda.to_cpu(x[0]) - cuda.to_cpu(x[1])
            # args = [[cuda.to_cpu(i) for i in inputs] for inputs in [x, gy, cgy]]
            # results = self.backward_cpu(*args)
            # del self.diff
            # t = [numpy.allclose(r, cuda.to_cpu(o), equal_nan=True) 
            #      for result,output in zip(results, outputs) 
            #      for r,o in zip(*[result, output])]
            # if not all(t):
            #     err = numpy.max([numpy.abs(r - cuda.to_cpu(o)) 
            #            for result,output in zip(results, outputs) 
            #            for r,o in zip(*[result, output])])
            #     print("\tWARNING in mse: max abs error: {}".format(err)) 
            #     # import pdb; pdb.set_trace()
            return outputs
        else:
            cuda.elementwise(
                '''{ctype}* gx, const {ctype}* x0, 
                   const {ctype}* x1, const {ctype}* gy, 
                   const {ctype}* coeff'''.format(ctype=self.ctype),
                '''gx[i]  = *gy  * (*coeff) * (x0[i] - x1[i])''',
                'mse_bwd')(gx, x0, x1, gy[0], coeff)
            outputs = (gx, -gx), (None, None)
            return outputs


def mean_squared_error(x0, x1):
    """Mean squared error function.

    This function computes mean squared error between two variables. The mean
    is taken over the minibatch. Note that the error is not scaled by 1/2.

    """
    return MeanSquaredError()(x0, x1)

