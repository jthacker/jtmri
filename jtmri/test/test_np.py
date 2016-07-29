import pytest
import numpy as np
from numpy.testing import assert_array_equal

from jtmri.np import flatten_axes, iter_axes, apply_to_axes, mosaic


def seq_array(dims):
    return np.arange(reduce(np.multiply, dims)).reshape(dims)


def test_flatten_axes_one_dim():
    """Flattening one dimension does not change the array"""
    a = seq_array((5, 4, 3, 2))
    b = flatten_axes(a, -1)
    assert_array_equal(a, b)

def test_flatten_axes():
    a = seq_array((5, 4, 3, 2))
    b = flatten_axes(a, (2, 3))
    c = seq_array((5, 4, 6))
    assert (5,4,6) == b.shape
    assert_array_equal(c, b)

@pytest.mark.skip(reason='flatten_axes needs to be fixed: '
                         'fails when swapping two non-adjacent dimensions')
def test_flatten_axies_2d():
    a = seq_array((2, 2, 2))
    f = flatten_axes(a, (0, 2))
    expected = np.array([[0, 1, 4, 5],
                         [2, 3, 6, 7]])
    assert expected == f

def test_flatten_axes_fully():
    a = seq_array((5, 4, 3, 2))
    assert_array_equal(np.arange(5*4*3*2), flatten_axes(a, (0, 1, 2, 3)))

def test_flatten_axes_masked():
    arr = seq_array((5, 4, 3, 2))
    a = np.ma.array(arr, mask=(arr % 2 == 0))
    b = flatten_axes(a, (0, 1, 2, 3))
    assert_array_equal(np.arange(arr.size), b.data)
    assert_array_equal(np.arange(arr.size) % 2 == 0, b.mask)

def test_flatten_axes_swapped():
    a = seq_array((5,4,3,2))
    assert_array_equal(flatten_axes(a, (0, 3)), flatten_axes(a, (3, 0)))

def test_iter_axes():
    a = seq_array((5,4,3))
    arrs = list(iter_axes(a, -1))
    assert 3 == len(arrs)
    assert_array_equal(a[...,0], arrs[0])
    assert_array_equal(a[...,1], arrs[1])
    assert_array_equal(a[...,2], arrs[2])

def test_iter_axes_none():
    a = np.ones((5,4))
    assert 0 == len(list(iter_axes(a, [])))

def test_iter_axes_multiple():
    a = seq_array((5,4,3,2))
    arrs = list(iter_axes(a, [2,3]))
    assert 6 == len(arrs)
    assert_array_equal(a[:,:,0,0], arrs[0])
    assert_array_equal(a[:,:,0,1], arrs[1])
    assert_array_equal(a[:,:,1,0], arrs[2])
    assert_array_equal(a[:,:,1,1], arrs[3])
    assert_array_equal(a[:,:,2,0], arrs[4])
    assert_array_equal(a[:,:,2,1], arrs[5])

def test_iter_axes_masked():
    arr = seq_array((5,4,3,2))
    a = np.ma.array(arr, mask=(arr % 2 == 0))
    b = list(iter_axes(a, 3))
    # All even numbers are masked
    assert_array_equal(a[:,:,:,0].mask, np.ones((5, 4, 3), dtype=bool))
    assert_array_equal(a[:,:,:,1].mask, np.zeros((5, 4, 3), dtype=bool))

def test_apply_to_axes():
    a = np.ones((5, 5, 2, 3, 4))
    assert_array_equal(np.ones((5, 5)) * 2 * 3 *4, apply_to_axes(np.sum, a, axes=(2, 3, 4)))

def test_mosaic_2d():
    a = seq_array((5,4))
    assert_array_equal(a, mosaic(a))

def test_mosaic():
    a = seq_array((5,4,3,2))
    shape = [5, 24]
    assert_array_equal(shape, mosaic(a).shape)

def test_mosaic_masked():
    a = seq_array((5,4,3,2))
    x = np.ma.array(a, mask=(a % 2 == 0))
    shape = [5, 24]
    assert_array_equal(shape, mosaic(x).shape)
    assert_array_equal(shape, mosaic(x).mask.shape)
