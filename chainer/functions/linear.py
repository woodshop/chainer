import math

import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    return x.reshape(x.shape[0], x.size // x.shape[0])


class Linear(function.Function):

    """Linear function (a.k.a. fully-connected layer or affine transformation).

    This function holds a weight matrix ``W`` and a bias vector ``b``.

    The weight matrix ``W`` has shape ``(out_size, in_size)``.
    This matrix is initialized with i.i.d. Gaussian samples, each of which has
    zero mean and deviation :math:`\sqrt{1/\\text{in_size}}`.
    The deviation is scaled by factor ``wscale`` if specified.

    The bias vector ``b`` is of size ``out_size``.
    Each element is initialized with the ``bias`` value.
    If ``nobias`` argument is set to True, then this function does not hold a
    bias vector.

    Let :math:`X` be an input matrix, and :math:`W, b` the weight matrix and
    the bias vector, respectively.
    Then, the output matrix :math:`Y` is computed by :math:`Y = XW^\\top + b`,
    where the addition by :math:`b` is broadcasted across the minibatch.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        bias (float): Initial bias value.
        nobias (bool): If True, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, then this
            function uses to initialize ``wscale``.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.

    .. note::

       This function accepts an input variable of a non-matrix array.
       In this case, the leading dimension is treated as the batch dimension,
       and the other dimensions are reduced to one dimension.

    """

    def __init__(self, in_size, out_size, wscale=1, bias=0, nobias=False,
                 initialW=None, initial_bias=None, cplx=None):
        self.W = None
        self.gW = None
        self.b = None
        self.gb = None
        if cplx is not None:
            self.cplx = cplx
            if cplx:
                self.dtype = numpy.complex64
                self.ctype = 'pycuda::complex<float>'
            else:
                self.dtype = numpy.float32
                self.ctype = 'float'
            
        if initialW is not None:
            assert initialW.shape == (out_size, in_size)
            self.W = initialW
        else:
             self.W = numpy.random.normal(
                0, wscale * math.sqrt(1. / in_size),
                (out_size, in_size)).astype(numpy.float32)
             if cplx:
                 self.W = self.dtype(
                     self.W + 1j * 
                     numpy.random.normal(0, wscale * math.sqrt(1. / in_size), 
                                         (out_size, in_size)))

        if isinstance(self.W, cuda.GPUArray):
            self.gW = cuda.empty_like(self.W)
        else:
            self.gW = numpy.empty_like(self.W)

        if initial_bias is not None:
            assert initial_bias.shape == (out_size,)
            self.b = initial_bias
        elif not nobias:
            if cplx:
                self.b = numpy.repeat(self.dtype(bias), out_size)
            else:
                self.b = numpy.repeat(numpy.float32(bias), out_size)

        if self.b is not None:
            if isinstance(self.b, cuda.GPUArray):
                self.gb = cuda.empty_like(self.b)
            else:
                self.gb = numpy.empty_like(self.b)

    @property
    def parameter_names(self):
        if self.b is None:
            return 'W',
        return 'W', 'b'

    @property
    def gradient_names(self):
        if self.gb is None:
            return 'gW',
        return 'gW', 'gb'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(
            x_type.dtype == self.dtype,
            x_type.ndim >= 2,
            (type_check.Variable(numpy.prod, 'prod')(x_type.shape[1:]) ==
             type_check.Variable(self.W.shape[1], 'W.shape[1]')),
        )

    def zero_grads(self):
        self.gW.fill(0)
        if self.gb is not None:
            self.gb.fill(0)

    def forward_cpu(self, x):
        x = _as_mat(x[0])
        Wx = x.dot(self.W.T)
        if self.b is not None:
            Wx += self.b
        return Wx,

    def forward_gpu(self, x):
        x = _as_mat(x[0])
        y = cuda.empty((x.shape[0], self.W.shape[0]), dtype=x.dtype)
        with cuda.using_cumisc():
            cuda.culinalg.dot(x, self.W, transb='T', out=y)
        if self.b is not None:
            cuda.elementwise(
                '{ctype}* y, {ctype}* b, int n_channel'.format(
                    ctype=self.ctype),
                'y[i] += b[i % n_channel]',
                'linear_bias')(y, self.b, self.b.size)
        return y,

    def backward_cpu(self, x, gy, cgy):
        _x = _as_mat(x[0])
        self.gW += gy[0].T.dot(_x)
        if self.gb is not None:
            self.gb += gy[0].sum(0)
        gx = gy[0].dot(self.W).reshape(x[0].shape)
        if self.cplx:
            cgx = cgy[0].dot(numpy.conj(self.W)).reshape(x[0].shape)
        else:
            cgx = None
        return (gx,), (cgx,)

    def backward_gpu(self, x, gy, cgy):
        # oldgW = self.gW.copy()
        # oldgb = self.gb.copy()
        _x = _as_mat(x[0])
        gx = cuda.empty_like(_x)
        cgx = cuda.empty_like(_x)
        with cuda.using_cumisc():
            cuda.culinalg.add_dot(gy[0], _x, self.gW, transa='T')
            # The following form is equivalent. However if this line
            # (and the respective line in the bias update)
            # is used, then the update_one method of the optimizer should 
            # be set to use a non-conjugate update:
            # cuda.culinalg.add_dot(cgy[0], _x.conj(), self.gW, transa='T')
            if self.gb is not None:
                self.gb += cuda.cumisc.sum(gy[0], 0)
                # self.gb += cuda.cumisc.sum(cgy[0], 0)
            cuda.culinalg.dot(gy[0], self.W, out=gx)
            if self.cplx:
                cuda.culinalg.dot(cgy[0], self.W.conj(), out=cgx)
        if self.cplx:
            outputs = (gx.reshape(x[0].shape),), (cgx.reshape(x[0].shape),)
        else:
            outputs = (gx.reshape(x[0].shape),), (None,)

        # args = [[cuda.to_cpu(i) for i in inputs] for inputs in [x, gy, cgy]]
        # self.W = cuda.to_cpu(self.W)
        # self.b = cuda.to_cpu(self.b)
        # newgW = self.gW.copy()
        # newgb = self.gb.copy()
        # self.gW = cuda.to_cpu(oldgW)
        # self.gb = cuda.to_cpu(oldgb)
        # results = self.backward_cpu(*args)
        # t = [numpy.allclose(r, cuda.to_cpu(o), equal_nan=True) 
        #      for result,output in zip(results, outputs) 
        #      for r,o in zip(*[result, output])]
        # t.extend([numpy.allclose(self.gW, cuda.to_cpu(newgW))])
        # t.extend([numpy.allclose(self.gb, cuda.to_cpu(newgb))])
        # if not all(t):
        #     err = [numpy.abs(r - cuda.to_cpu(o)) 
        #            for result,output in zip(results, outputs) 
        #            for r,o in zip(*[result, output])]
        #     err.extend([numpy.abs(self.gW - cuda.to_cpu(newgW))])
        #     err.extend([numpy.abs(self.gb - cuda.to_cpu(newgb))])
        #     err = numpy.max([numpy.max(e) for e in err])
        #     print("\tWARNING in linear: max abs error: {}".format(err)) 
        #     import pdb; pdb.set_trace()
        # self.W = cuda.to_gpu(self.W)
        # self.b = cuda.to_cpu(self.b)
        # self.gW = newgW
        # self.gb = newgb

        return outputs

