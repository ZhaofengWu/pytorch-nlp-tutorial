"""
This is a very brief introduction to numpy. Some examples are adapted from [here](https://cs231n.github.io/python-numpy-tutorial/#numpy).
For a much more comprehensive introduction, see the [official quickstart guide](https://numpy.org/devdocs/user/quickstart.html).

What is numpy? In [it's own words](https://numpy.org/devdocs/user/whatisnumpy.html):
"NumPy is the fundamental package for scientific computing in Python. It is a Python library that
 provides a multidimensional array object, various derived objects (such as masked arrays and
 matrices), and an assortment of routines for fast operations on arrays, including mathematical,
 logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear
 algebra, basic statistical operations, random simulation and much more."

In short, it makes multidimensional array manipulation (our data often comes in this format) easy
and fast.
"""

import numpy as np


## Array Creation

# We can create numpy arrays from python lists, and index them like regular python lists.
# Let's create a one-dimensional array first.
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)
print(a)
print(a.tolist())  # This converts an numpy array back to a normal python list
print(a[0], a[1], a[2])
a[0] = 5
print(a)

# Now, let's create a two-dimensional array.
b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(b.shape)
print(b.ndim)
print(b[0, 0], b[0, 1], b[1, 0])

# And an array with a different data type (dtype).
c = np.array([1, 2, 3], dtype=np.float)
# Equivalently, numpy automatically deduces the dtype based on the element type of the list.
d = np.array([1.0, 2.0, 3.0])
print(a.dtype, c.dtype, d.dtype)
# We can change the dtype of an numpy array at any time.
print(a, a.astype(np.float))

# numpy also provides some convenient methods to create arrays.
print(np.empty((2, 2)))  # No initialization
print(np.zeros((2, 2)))
print(np.ones((2, 2)))
print(np.full((2, 2), 7))
print(np.eye(4))
print(np.arange(10, 30, 5))  # Like the python built-in range function
print(np.random.random((2, 2)))


## Array Indexing And Shape

# numpy arrays support all fancy indexing methods that python lists do...
print(a)
print(a[1:])
print(a[1:2])
print(a[::-1])

# ... and more! Multidimensional arrays can be indexed with a tuple of indices, each indexing into
# one dimension.
print(b)
print(b[1, :2])
# Or with another numpy array. Note that this is equivalent to b[1:3], not b[1, 2].
print(b[np.array([1, 2])])
# Or with a boolean mask, which will flatten the original array.
mask = np.array(
    [[True, True, False], [False, True, False], [False, False, False], [False, True, False]]
)
print(b[mask])
# Or changing the orignal array with any of these indexing methods.
print(b > 4)
b[b > 4] = 0
print(b)

# numpy arrays reside in an underlying contiguous chunk of memory. Therefore, we can reshape the
# array with (most of the time) no overhead.
print(b.shape)
print(b.reshape((3, 4)).shape)
# We can use -1 when a size of a dimension can be deduced from the other dimensions.
print(b.reshape((-1, 4)).shape)
print(b.reshape((-1,)).shape)
# And due to the shared memory, changing the data of a reshaped array changes the original too.
print(b)
c = b.reshape((-1,))
c[0] = 99
print(b)


## Array Math

# numpy supports many operations on arrays. Many of these are elementwise.
d = np.array([1, 2, 3], dtype=np.float)
e = np.array([4, 5, 6], dtype=np.float)
print(d + e)
print(d - e)
print(d * e)
print(d / e)
print(np.sqrt(d))
print(np.sin(d))

# We can also perform operations such as dot product, matrix multiplication, transpose, etc.
print(d.dot(e))
f = np.arange(12).reshape(3, 4)
print(f)
print(f @ f.T)  # @ is matrix multiplication, .T is transpose

# There are also reduction operations that can either applied to the entire array or along a
# specific axis. Note that for many functions, we can either do np.func(array) or array.func().
print(f)
print(np.sum(f))
print(f.sum())
print(f.sum(axis=0))
print(f.sum(axis=1))


## Array broadcasting

# One important feature of numpy that will make our life a lot easier is that we can perform
# operations on arrays of different dimensions or sizes. This is called "broadcasting" where numpy
# pads the array with fewer dimensions or sizes before the operation. The rules are quite complex,
# see [the official documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).
# We will only show some simple examples.
g = f.reshape(4, 3)  # This is to satisfy the broadcasting rules
print(d)
print(g)
print(d.shape, g.shape)
print(d + g)

# We cannot do this on input that does not satisfy the broadcasting rules.
print(d.shape, f.shape)
try:
    print(d + f)
except Exception:
    import traceback
    print(traceback.format_exc())

# However, we can manually pad the arrays to make this work.
h = d.reshape((3, -1))
print(h.shape, f.shape)
print(h + f)
