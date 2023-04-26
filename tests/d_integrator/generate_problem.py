import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

NSTATES= 4
NINPUTS = 2
NHORIZON = 51  # = num of horizon controls

A = np.array([[1, 0, 0.1, 0],[0, 1, 0, 0.1],[0, 0, 1.0, 0],[0, 0, 0, 1.0]])
B = np.array([[0.005, 0], [0, 0.005], [0.1, 0], [0, 0.1]])

x0 = np.array([5, 7, 2, -1.4])
xg = np.array([0, 0, 0, 0.0])
umin = np.array([-5, -5])
umax = np.array([5, 5])

Q = np.zeros((NSTATES, NSTATES))
np.fill_diagonal(Q, 0.1)
R = np.zeros((NINPUTS, NINPUTS))
np.fill_diagonal(R, 0.1)
Qf = np.zeros((NSTATES, NSTATES))
np.fill_diagonal(Qf, 100*0.1)

xhist = np.zeros((NSTATES,NHORIZON+1))
uhist = np.zeros((NINPUTS,NHORIZON))
xhist[:, 0] = np.copy(x0)
print(xhist[:, 0])
# Cost
U_pick = np.kron(np.eye(NHORIZON), np.hstack([np.eye(NINPUTS), np.zeros((NINPUTS, NSTATES))]))
H_block = np.block([[R, np.zeros((NINPUTS, NSTATES))],
                    [np.zeros((NSTATES, NINPUTS)), Q]])
H = np.kron(np.eye(NHORIZON), H_block)
b = np.zeros(NHORIZON*(NSTATES+NINPUTS))

# Constraints
lb = np.zeros(NHORIZON*(NSTATES+NINPUTS))
ub = np.zeros(NHORIZON*(NSTATES+NINPUTS))

lb[0:NSTATES] = -A@xhist[:,0]
ub[0:NSTATES] = -A@xhist[:,0]  # equality constraints => dynamics

C = np.kron(np.eye(NHORIZON), np.hstack([B, -np.eye(NSTATES)]))

lb[np.shape(C)[0]:np.shape(C)[0]+NINPUTS] = np.copy(umin)
ub[np.shape(C)[0]:np.shape(C)[0]+NINPUTS] = np.copy(umax)

for k in range(1,NHORIZON-1):
  C[(k*NSTATES):(k*NSTATES+NSTATES), (k*(NSTATES+NINPUTS)-NSTATES):((k+1)*(NSTATES+NINPUTS)-NSTATES)] = np.hstack([A, B])
  lb[k*NINPUTS+np.shape(C)[0]:k*NINPUTS+np.shape(C)[0]+NINPUTS] = np.copy(umin)
  ub[k*NINPUTS+np.shape(C)[0]:k*NINPUTS+np.shape(C)[0]+NINPUTS] = np.copy(umax)


D = np.vstack([C, U_pick])
H[-NSTATES:,-NSTATES:] = np.copy(Qf)

P = sparse.triu(H, format='csc')
q = b

A = sparse.csc_matrix(D)
l = lb
u = ub

n = P.shape[0]
m = A.shape[0]

# Generate problem data
cu.generate_problem_data(P, q, A, l, u, 'd_integrator')
