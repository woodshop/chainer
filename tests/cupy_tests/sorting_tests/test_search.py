import unittest

from cupy import testing


@testing.gpu
class TestSearch(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_all(self, xpy, dtype):
        a = testing.shaped_random((2, 3), xpy, dtype)
        return a.argmax()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis_large(self, xpy, dtype):
        a = testing.shaped_random((3, 1000), xpy, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis0(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmax(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis1(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmax(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmax_axis2(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmax(axis=2)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_all(self, xpy, dtype):
        a = testing.shaped_random((2, 3), xpy, dtype)
        return a.argmin()

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis_large(self, xpy, dtype):
        a = testing.shaped_random((3, 1000), xpy, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis0(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmin(axis=0)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis1(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmin(axis=1)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose()
    def test_argmin_axis2(self, xpy, dtype):
        a = testing.shaped_random((2, 3, 4), xpy, dtype)
        return a.argmin(axis=2)
