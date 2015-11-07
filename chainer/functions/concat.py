import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

_args = 'const {ctype}* x, {ctype}* y, int cdimx, int cdimy, int rdim, int coffset'
_preamble = '''
#define COPY(statement) \
    int l   = i / (rdim * cdimx);  \
    int c   = i / rdim % cdimx + coffset;  \
    int r   = i % rdim;  \
    int idx = r + rdim * (c + cdimy * l);  \
    statement;
'''


class Concat(function.Function):

    """Concatenate multiple tensors towards specified axis."""

    # concat along the channel dimension by default
    def __init__(self, axis=1):
        self.axis = axis

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() > 0)
        type_check.expect(in_types[0].ndim >
                          type_check.Variable(self.axis, 'axis'))

        ndim = in_types[0].ndim.eval()
        for i in range(1, in_types.size().eval()):
            type_check.expect(
                in_types[0].dtype == in_types[i].dtype,
                in_types[0].ndim == in_types[i].ndim,
            )
            for d in range(0, ndim):
                if d == self.axis:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])
        self.cplx = in_types[0].dtype.eval() == numpy.complex64


    def forward_cpu(self, xs):
        return numpy.concatenate(xs, axis=self.axis),

    def forward_gpu(self, xs):
        # TODO(beam2d): Unify the process into a single kernel.
        shape = list(xs[0].shape)
        for x in xs[1:]:
            shape[self.axis] += x.shape[self.axis]
        shape = tuple(shape)
        self.shape = shape

        y = cuda.empty(shape, dtype=xs[0].dtype)
        self.cdimy = y.shape[self.axis]
        self.rdim = numpy.prod(shape[self.axis + 1:], dtype=int)

        coffset = 0
        kernel = cuda.elementwise(
            _args.format(ctype=self.ctype), 'COPY(y[idx] = x[i])', 
            'concat_fwd', 
            preamble=_preamble)
        for x in xs:
            cdimx = x.shape[self.axis]
            kernel(x, y, cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx

        return y,

    def backward_cpu(self, xs, gy, cgy):
        sizes = numpy.array([x.shape[self.axis] for x in xs[:-1]]).cumsum()
        return (numpy.split(gy[0], sizes, axis=self.axis),
                numpy.split(cgy[0], sizes, axis=self.axis))

    def backward_gpu(self, xs, gy, cgy):
        gxs = tuple(cuda.empty_like(x) for x in xs)
        cgxs = tuple(cuda.empty_like(x) for x in xs)

        kernel = cuda.elementwise(
            _args.format(ctype=self.ctype), 'COPY(x[i] = y[idx])', 
            'concat_bwd', preamble=_preamble)
        coffset = 0
        for gx in gxs:
            cdimx = gx.shape[self.axis]
            kernel(gx, gy[0], cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx
        coffset = 0
        for cgx in cgxs:
            cdimx = cgx.shape[self.axis]
            kernel(cgx, cgy[0], cdimx, self.cdimy, self.rdim, coffset)
            coffset += cdimx
        outputs = gxs, cgxs

        ### THIS CHECKS OUT
        # args = [[cuda.to_cpu(i) for i in inputs] for inputs in [xs, gy, cgy]]
        # results = self.backward_cpu(*args)
        # t = [numpy.allclose(r, cuda.to_cpu(o), equal_nan=True) 
        #      for result,output in zip(results, outputs) 
        #      for r,o in zip(*[result, output])]
        # if not all(t):
        #     err = numpy.max([numpy.abs(r - cuda.to_cpu(o)) 
        #            for result,output in zip(results, outputs) 
        #            for r,o in zip(*[result, output])])
        #     print("\tWARNING in concat: max abs error: {}".format(err)) 
        #     import pdb; pdb.set_trace()

        return outputs


def concat(xs, axis=1):
    """Concatenates given variables along an axis.

    Args:
        xs (tuple of Variables): Variables to be concatenated.
        axis (int): Axis that the input arrays are concatenated along.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return Concat(axis=axis)(*xs)
