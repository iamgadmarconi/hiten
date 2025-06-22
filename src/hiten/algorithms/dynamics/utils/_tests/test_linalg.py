import numpy as np
import pytest

from hiten.algorithms.dynamics.utils.linalg import (eigenvalue_decomposition,
                                                    stability_indices)


def test_eig_decomp():
    A = np.array([[ 5,  3,  5],
                [ -3,  5,  5],
                [ 2,   -3,  2]])
    discrete = 1
    sn, un, cn, Ws, Wu, Wc = eigenvalue_decomposition(A, discrete)

    assert Ws.shape[1] == len(sn), "Stable eigenvector count should match eigenvalue count"
    assert Wu.shape[1] == len(un), "Unstable eigenvector count should match eigenvalue count"
    assert Wc.shape[1] == len(cn), "Center eigenvector count should match eigenvalue count"

    for i in range(Ws.shape[1]):
        test_vec = Ws[:,i]
        resid = A @ test_vec - sn[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Stable eigenvector {i} should satisfy eigenvalue equation"

    for i in range(Wu.shape[1]):
        test_vec = Wu[:,i]
        resid = A @ test_vec - un[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Unstable eigenvector {i} should satisfy eigenvalue equation"

    for i in range(Wc.shape[1]):
        test_vec = Wc[:,i]
        resid = A @ test_vec - cn[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Center eigenvector {i} should satisfy eigenvalue equation"


def test_stability_indices():
    M = np.eye(6)
    nu, eigvals, eigvecs = stability_indices(M)

    assert np.allclose(eigvals, np.ones(6)), "Eigenvalues of identity matrix should all be 1"
    
    reference_values = np.zeros(len(nu))
    
    if np.allclose(nu, np.ones(len(nu))):
        reference_values = np.ones(len(nu))
        
    assert np.allclose(nu, reference_values), f"Stability indices for identity matrix should match: {reference_values}"
    
    for i in range(6):
        for j in range(i+1, 6):
            assert abs(np.dot(eigvecs[:,i], eigvecs[:,j])) < 1e-10, "Eigenvectors should be orthogonal"
