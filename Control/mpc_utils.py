import numpy as np

def build_pred_mats_with_bias(A, B, b, N):
    """
    Constructs prediction matrices with bias term for MPC.

    :param A: State transition matrix.
    :param B: Control input matrix.
    :param b: Bias vector.
    :param N: Prediction horizon.
    :return: Phi (free response matrix), Gamma (Control response matrix), d (accumulated bias vector).
    """
    r = A.shape[0]
    m = B.shape[1]
    Phi = np.zeros((r*N, r))
    Gamma = np.zeros((r*N, m*N))
    d = np.zeros((r*N,))

    for i in range(N):
        Ai = np.linalg.matrix_power(A, i+1)
        Phi[i*r:(i+1)*r, :] = Ai

        for j in range(i+1):
            Aij = np.linalg.matrix_power(A, i-j)
            Gamma[i*r:(i+1)*r, j*m:(j+1)*m] = Aij@B
        bias_sum = np.zeros((r,))
        for j in range(i+1):
            bias_sum += np.linalg.matrix_power(A, j)@b
        d[i*r:(i+1)*r] = bias_sum

    return Phi, Gamma, d