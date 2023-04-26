import numpy as np
from scipy import sparse
import utils.codegen_utils as cu

NSTATES= 12
NINPUTS = 4
NHORIZON = 20  # = num of horizon controls

A = np.array([
  [1.000000,0.000000,0.000000,0.000000,0.003924,0.000000,0.020000,0.000000,0.000000,0.000000,0.000013,0.000000],
  [0.000000,1.000000,0.000000,-0.003924,0.000000,0.000000,0.000000,0.020000,0.000000,-0.000013,0.000000,0.000000],
  [0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.020000,0.000000,0.000000,0.000000],
  [0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.010000,0.000000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.010000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.010000],
  [0.000000,0.000000,0.000000,0.000000,0.392400,0.000000,1.000000,0.000000,0.000000,0.000000,0.001962,0.000000],
  [0.000000,0.000000,0.000000,-0.392400,0.000000,0.000000,0.000000,1.000000,0.000000,-0.001962,0.000000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000,0.000000],
  [0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000],
])

B = np.array([
  [-0.000019,-0.000001,0.000019,0.000001],
  [-0.000001,-0.000019,0.000001,0.000019],
  [0.000981,0.000981,0.000981,0.000981],
  [0.001264,0.029044,-0.001493,-0.028815],
  [-0.029414,-0.001057,0.028771,0.001700],
  [0.004771,-0.003644,0.001265,-0.002392],
  [-0.003847,-0.000138,0.003763,0.000222],
  [-0.000165,-0.003799,0.000195,0.003769],
  [0.098100,0.098100,0.098100,0.098100],
  [0.252748,5.808852,-0.298680,-5.762921],
  [-5.882783,-0.211410,5.754175,0.340018],
  [0.954290,-0.728857,0.252942,-0.478376],
])

x0 = np.array([-0.5, 0.5, -0.5, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
xg = np.zeros_like(x0)
umin = np.array([-0.5, -0.5, -0.5, -0.5])
umax = np.array([0.5, 0.5, 0.5, 0.5])

Q = np.zeros((NSTATES, NSTATES))
np.fill_diagonal(Q, [1, 1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
R = np.zeros((NINPUTS, NINPUTS))
np.fill_diagonal(R, 1e-1)
Qf = 10*Q

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
cu.generate_problem_data(P, q, A, l, u, 'quadrotor')
