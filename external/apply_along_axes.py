import itertools
import numpy as np
from vorpy.index_map import *

def apply_along_axes (func, input_axis_v, input_array_v, *args, output_axis_v=None, func_output_shape=None, **kwargs):
    """
    This function is a generalization of numpy.apply_along_axis.

    TODO: Add a parameter for the assignment destination, so that no temporaries are created.

    Let A = (0,1,...,N-1), where N is len(input_array_v[0].shape).  input_axis_v should be* a nonempty
    subsequence of A; this means that 0 < len(input_axis_v) <= len(A), each element of input_axis_v
    must be an element of the sequence A, and input_axis_v must be strictly increasing (as A is).

    *Note that input_axis_v may contain values in the range [-N,0), in which case, N is added to
    each negative element to bring them all within the range [0,N) as valid axis indices.  For example,
    this means that -1 addresses the last axis, -2 addresses the second-to-last axis, etc.

    TODO: Allow the base case where input_array_v[0].shape = () and input_axis_v = [].

    Let I be the complementary subsequence of input_axis_v, meaning that I is the sequence consisting
    of elements of A not found in input_axis_v, and I is strictly increasing (as A is).

    func is then called on each slice of each element of input_array_v where the indices specified by I
    are fixed and the indices specified by input_axis_v are free.  The shape of the return value of func
    on the first call is used to determine the shape of the return value of apply_along_axes, or if
    func_output_shape is not None, then func_output_shape is assumed to be the shape of the return
    value of func.

    Let S denote the shape of the return value of func (which may be specified by func_output_shape as
    specified above).  Let B = (0,1,...,M-1), where M is len(I)+len(S).  The return value of
    apply_along_axes will be a numpy.ndarray having M indices.

    If output_axis_v is not None, it must be** a subsequence of B, specifying which axes of the
    return value will be used to index the output of func.  If output_axis_v is None, then it
    will be assumed to have value (L,L+1,...,M-1), where L = len(I), i.e. the output axes will
    be the last axes of the return value.

    **Note that, just as input_axis_v, output_axis_v may contain values in the range [-N,0), in which
    case, N is added to bring each negative element to bring them all within the range [0,N).

    All extra args and kwargs will be passed through to calls to func.

    Note that the single-element input_axis_v special-case call

        apply_along_axes(func, (i,), (input_array,), ...)

    should be equivalent to the standard numpy call

        numpy.apply_along_axis(func, i, input_array, ...)
    """

    assert len(input_axis_v) > 0, 'input_axis_v (which is {0}) must be a nonempty subsequence of (0,1,...,{1})'.format(input_axis_v, len(input_array.shape)-1)
    assert len(input_array_v) > 0, 'input_array_v (which has length {0}) must have length > 0'.format(len(input_array_v))
    assert len(frozenset(input_array.shape for input_array in input_array_v)) == 1, 'each element of input_array_v must have the same shape, but encountered shapes {0}'.format([input_array.shape for input_array in input_array_v])
    assert len(frozenset(input_array.dtype for input_array in input_array_v)) == 1, 'each element of input_array_v must have the same dtype, but encountered dtypes {0}'.format([input_array.dtype for input_array in input_array_v])

    input_array_shape = input_array_v[0].shape
    input_array_dtype = input_array_v[0].dtype
    N = len(input_array_shape)
    A = np.arange(N)
    # Note that the length of the complementary subsequence of N in A is N-len(input_axis_v).

    assert all(-N <= input_axis < N for input_axis in input_axis_v), 'input_axis_v (which is {0}), must contain values in the range of [-N,N) (where N = {1})'.format(input_axis_v, N)

    normalized_input_axis_v = np.fromiter((input_axis if input_axis >= 0 else input_axis+N for input_axis in input_axis_v), np.int, len(input_axis_v))

    assert is_subsequence_of_nondecreasing_sequence(normalized_input_axis_v, A), 'normalized_input_axis_v (which is {0}) must be a nonempty subsequence of [0,{1}) (right endpoint excluded)'.format(normalized_input_axis_v, len(input_array_shape))

    is_not_normalized_input_axis_v = np.array([axis_index not in normalized_input_axis_v for axis_index in A])
    non_normalized_input_axis_v = np.array([axis_index for axis_index in A if is_not_normalized_input_axis_v[axis_index]])
    input_iterator_v = [range(input_array_shape[axis_index]) if is_not_normalized_input_axis_v[axis_index] else [slice(None)] for axis_index in A]
    # print('is_not_normalized_input_axis_v = {0}'.format(is_not_normalized_input_axis_v))
    # print('non_normalized_input_axis_v = {0}'.format(non_normalized_input_axis_v))
    # print('input_iterator_v = {0}'.format(input_iterator_v))

    # If func_output_shape is not specified, then derive it.
    if func_output_shape is None:
        func_output_shape = np.shape(func(*tuple(input_array[tuple(0 if axis_index not in normalized_input_axis_v else slice(None) for axis_index in A)] for input_array in input_array_v), *args, **kwargs))
    # print('func_output_shape = {0}'.format(func_output_shape))

    # B = np.arange(N-len(normalized_input_axis_v)+len(func_output_shape))
    B = np.arange(len(non_normalized_input_axis_v)+len(func_output_shape))
    M = len(B)
    # print('B = {0}'.format(B))

    # If output_axis_v is None, then it will be assumed to be the trailing axes of the output.
    if output_axis_v is None:
        output_axis_v = np.arange(-len(func_output_shape), 0)
    # print('normalized_input_axis_v = {0}'.format(normalized_input_axis_v))
    # print('output_axis_v = {0}'.format(output_axis_v))

    assert len(func_output_shape) == len(output_axis_v), 'func_output_shape (which is {0}) must have same number of elements as output_axis_v (which is {1})'.format(func_output_shape, output_axis_v)

    normalized_output_axis_v = np.fromiter((output_axis if output_axis >= 0 else output_axis+M for output_axis in output_axis_v), np.int, len(output_axis_v))

    assert is_subsequence_of_nondecreasing_sequence(normalized_output_axis_v, B), 'normalized_output_axis_v (which is {0}) must be a subsequence of [0,{1}) (right endpoint excluded)'.format(normalized_output_axis_v, len(B))

    is_not_output_axis_v = [axis_index not in normalized_output_axis_v for axis_index in B]
    # print('is_not_output_axis_v = {0}'.format(is_not_output_axis_v))
    non_output_axis_v = [axis_index for axis_index in B if is_not_output_axis_v[axis_index]]
    # print('non_output_axis_v = {0}'.format(non_output_axis_v))
    non_output_axis_inv_v = index_map_inverse(non_output_axis_v, len(B))
    # print('non_output_axis_inv_v = {0}'.format(non_output_axis_inv_v))
    output_axis_inv_v = index_map_inverse(normalized_output_axis_v, len(B))
    # print('output_axis_inv_v = {0}'.format(output_axis_inv_v))
    # print('compose_index_maps(non_normalized_input_axis_v, non_output_axis_inv_v) = {0}'.format(compose_index_maps(non_normalized_input_axis_v, non_output_axis_inv_v)))
    output_iterator_v = [range(input_array_shape[non_normalized_input_axis_v[non_output_axis_inv_v[axis_index]]]) if is_not_output_axis_v[axis_index] else [slice(None)] for axis_index in B]
    # print('output_iterator_v = {0}'.format(output_iterator_v))

    retval = np.ndarray(tuple(input_array_shape[non_normalized_input_axis_v[non_output_axis_inv_v[axis_index]]] if is_not_output_axis_v[axis_index] else func_output_shape[output_axis_inv_v[axis_index]] for axis_index in B), dtype=input_array_dtype)
    # print('retval.shape = {0}'.format(retval.shape))
    for input_I,output_I in zip(itertools.product(*input_iterator_v), itertools.product(*output_iterator_v)):
        # print('input_I = {0}, output_I = {1}'.format(input_I, output_I))
        retval[output_I] = func(*tuple(input_array[input_I] for input_array in input_array_v), *args, **kwargs)

    # print('')
    return retval
