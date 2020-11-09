import numpy as np
from sympy import Matrix


def generate_reparameterisation_matrix(S: np.array):
    """Get a reparameterisation matrix for a stoichiometric matrix.

    Uses the method from this paper:

    - Alberty, R. A. (1991). Equilibrium compositions of solutions of
      biochemical species and heats of biochemical reactions. Proceedings of
      the National Academy of Sciences of the United States of America, 88(8),
      3268–3271.

    """
    ST_rref, leading_one_cols = Matrix(S.T).rref()
    ST_rref = np.array(ST_rref).astype(np.float64)
    R = np.eye(S.shape[0])
    for i, col in enumerate(leading_one_cols):
        R[col] = ST_rref[i]
    return R
