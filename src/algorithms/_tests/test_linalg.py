import numpy as np
import pytest

from algorithms.linalg import eigenvalue_decomposition, stability_indices




def test_eig_decomp():
    # Build a sample matrix A, same as the MATLAB example
    A = np.array([[ 5,  3,  5],
                [ -3,  5,  5],
                [ 2,   -3,  2]])
    # Call the eig_decomp function
    discrete = 1
    sn, un, cn, Ws, Wu, Wc = eigenvalue_decomposition(A, discrete)

    # Test dimensions
    assert Ws.shape[1] == len(sn), "Stable eigenvector count should match eigenvalue count"
    assert Wu.shape[1] == len(un), "Unstable eigenvector count should match eigenvalue count"
    assert Wc.shape[1] == len(cn), "Center eigenvector count should match eigenvalue count"

    # Verify that A * w_s ~ sn(i) * w_s
    for i in range(Ws.shape[1]):
        test_vec = Ws[:,i]
        resid = A @ test_vec - sn[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Stable eigenvector {i} should satisfy eigenvalue equation"

    # Repeat for unstable eigenvectors if any exist
    for i in range(Wu.shape[1]):
        test_vec = Wu[:,i]
        resid = A @ test_vec - un[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Unstable eigenvector {i} should satisfy eigenvalue equation"

    # Repeat for center eigenvectors if any exist
    for i in range(Wc.shape[1]):
        test_vec = Wc[:,i]
        resid = A @ test_vec - cn[i]*test_vec
        assert np.linalg.norm(resid) < 1e-10, f"Center eigenvector {i} should satisfy eigenvalue equation"


def test_stability_indices():
    M = np.eye(6)
    nu, eigvals, eigvecs = stability_indices(M)

    # For identity matrix, all eigenvalues should be 1
    assert np.allclose(eigvals, np.ones(6)), "Eigenvalues of identity matrix should all be 1"
    
    # Stability indices should be all zeros for identity matrix
    assert np.allclose(nu, np.zeros(6)), "Stability indices for identity matrix should be zeros"
    
    # Eigenvectors should be orthogonal
    for i in range(6):
        for j in range(i+1, 6):
            assert abs(np.dot(eigvecs[:,i], eigvecs[:,j])) < 1e-10, "Eigenvectors should be orthogonal"