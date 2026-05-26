import numpy as np
import scipy.sparse as sp
import osqp
from mpc_utils import build_pred_mats_with_bias

class KoopmanMPC:
    def __init__(self, latent_dim, ctrl_dim, horizon, Qz, Ru, umin, umax):
        self.r = latent_dim
        self.m = ctrl_dim
        self.N_h = horizon
        self.Qz = Qz
        self.Ru = Ru
        self.umin = umin
        self.umax = umax
        self.prob = osqp.OSQP()
        self._setup_done = False

        self.A = None
        self.B = None
        self.b = None

    def setup_form(self, A, B, b):
        self.A = A
        self.B = B
        self.b = b
        
        N = self.N_h
        r = self.r
        m = self.m
        Phi, Gamma, d = build_pred_mats_with_bias(A, B, b, N)
        Qbar = sp.block_diag([sp.csc_matrix(self.Qz) for _ in range(N)], format='csc')
        Rbar = sp.block_diag([sp.csc_matrix(self.Ru) for _ in range(N)], format='csc')

        H = 2*(Gamma.T@(Qbar@Gamma)+Rbar)
        H = 0.5*(H+H.T)
        P = sp.csc_matrix(H)

        A_cons = sp.eye(m*N, format='csc')
        l = np.kron(np.ones(N), self.umin)
        u = np.kron(np.ones(N), self.umax)

        self.Phi = Phi
        self.Gamma = Gamma
        self.Qbar = Qbar
        self.d = d

        self.prob.setup(P=P, q=np.zeros((m*N,)), A=A_cons, l=l, u=u, verbose=False, warm_start=True, polish=True)
        self._setup_done = True

    def solve(self, z0, z_ref_seq):
        z_ref_stack = z_ref_seq.reshape(-1)
        z_free = self.Phi@z0 + self.d
        e = z_free - z_ref_stack
        q = 2*(self.Gamma.T@(self.Qbar@e))
        self.prob.update(q=q)
        res = self.prob.solve()

        if res.info.status_val not in (1, 2):
            return np.zeros((self.m,))
        return res.x[:self.m]