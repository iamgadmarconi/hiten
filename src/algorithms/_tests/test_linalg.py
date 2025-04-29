import numpy as np

from algorithms.linalg import eigenvalue_decomposition, stability_indices




def test_eig_decomp():
    # 1) Build a sample matrix A, same as the MATLAB example
    # We'll use a diagonal for clarity: [0.9, 1.1, 1.0, 1.0]
    A = np.array([[ 5,  3,  5],
                [ -3,  5,  5],
                [ 2,   -3,  2]])
    # 2) Call the eig_decomp function
    discrete = 1
    sn, un, cn, Ws, Wu, Wc = eigenvalue_decomposition(A, discrete)

    # 3) Print the results
    print("Stable eigenvalues:", sn)
    print("Unstable eigenvalues:", un)
    print("Center eigenvalues:", cn)

    print("Stable eigenvectors:", Ws)
    print("Unstable eigenvectors:", Wu)
    print("Center eigenvectors:", Wc)

    print("Stable subspace dimension:", Ws.shape[1])
    print("Unstable subspace dimension:", Wu.shape[1])
    print("Center subspace dimension:", Wc.shape[1])

    # 4) Optional: verify that A * w_s ~ sn(i) * w_s, etc.
    # For stable eigenvectors:
    for i in range(Ws.shape[1]):
        test_vec = Ws[:,i]
        check_resid = A @ test_vec - sn[i]*test_vec
        print(f"Ws vector {i} residue norm:", np.linalg.norm(check_resid))

    print("test_eig_decomp completed successfully.")


def test_stability_indices():
    M = np.eye(6)
    nu, eigvals, eigvecs = stability_indices(M)

    print("Stability indices:", nu)
    print("Eigenvalues:", eigvals)
    print("Eigenvectors:", eigvecs)

    print("test_stability_indices completed successfully.")


if __name__ == "__main__":
    test_eig_decomp()
    test_stability_indices()