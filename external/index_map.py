import numpy as np

def is_nondecreasing_sequence (seq):
    """Not a publicly offered function."""
    return all(s0 <= s1 for (s0,s1) in zip(seq[:-1],seq[1:]))

def is_subsequence_of_nondecreasing_sequence (subseq, seq):
    """Not a publicly offered function."""
    assert is_nondecreasing_sequence(seq), 'seq is not nondecreasing'

    subseq_index = seq_index = 0
    while True:
        # If we've matched all elements of subseq, then we have a subsequence.
        if subseq_index == len(subseq):
            return True
        # This condition is guaranteed to occur at some point, guaranteeing that the function will return.
        elif seq_index == len(seq):
            return False

        # If the values at the cursors match, then so far so good, increment both cursors.
        if subseq[subseq_index] == seq[seq_index]:
            subseq_index += 1
        # Increment the sequence read cursor regardless.
        seq_index += 1

def compose_index_maps (*index_vv):
    """
    A list is a way to represent a map.  Assume list L has N elements.  Then L can be considered to
    map the elements of the sequence (0,1,...,N-1) to the elements of the sequence (L[0],L[1],...,L[N-1])
    respectively.

    This function composes a sequence of lists-as-functions, returning a list which represents
    the composition.
    """

    for domain_index_v,codomain_index_v in zip(reversed(index_vv[1:]),reversed(index_vv[:-1])):
        assert all(0 <= domain_index < len(codomain_index_v) for domain_index in domain_index_v), 'domain error; elements of index list {0} must map into the index range [0,...,{1}) (right endpoint excluded), which index the list {2}'.format(domain_index_v, len(codomain_index_v), codomain_index_v)

    current = np.arange(len(index_vv[-1]))
    for index_v in reversed(index_vv):
        current[:] = [index_v[i] for i in current]
    return current

def index_map_inverse (index_map_v, index_count):
    """
    Let index_map_inverse_v denote the return value of this function.  It will satisfy:

        np.all(compose_index_maps(index_map_inverse_v, index_map_v) == np.arange(len(index_map_v)))
    """
    index_map_list_v = list(index_map_v) if type(index_map_v) is not list else index_map_v
    assert len(frozenset(index_map_v)) == len(index_map_v), 'index_map_v (which is {0}) must have nonrepeating elements (i.e. it must be injective as a map)'.format(index_map_v)
    assert all(0 <= i < index_count for i in index_map_v), 'index_map_v (which is {0}) must have elements in the range [0,{1}) (right endpoint excluded)'.format(index_map_v, index_count)
    return np.array([index_map_list_v.index(i) if i in index_map_v else 0 for i in range(index_count)])
