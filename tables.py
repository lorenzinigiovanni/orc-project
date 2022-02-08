import matplotlib.pyplot as plt
import numpy as np

# Number of discretization steps for the joint angle q
NQ=51
# Number of discretization steps for the joint velocity v
NV=51

# Discretization resolution for joint angle
DQ = 2 * np.pi / NQ
# Discretization resolution for joint velocity
DV = 2 * 5 / NV 


def d2cq(iq):
    iq = np.clip(iq, 0, NQ - 1)
    return iq * DQ - np.pi + 0.5 * DQ


def d2cv( iv):
    iv = np.clip(iv, 0, NV - 1) - (NV - 1) / 2
    return iv * DV


# Plot the given Value table V 
def plot_V_table(V):
    Q, DQ = np.meshgrid([d2cq(i) for i in range(NQ)], [d2cv(i) for i in range(NV)])
    plt.pcolormesh(Q, DQ, V.reshape((NV, NQ)), cmap=plt.cm.get_cmap('Blues'))
    plt.colorbar()
    plt.title('V table')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()


# Plot the given policy table pi
def plot_policy(pi):
    Q, DQ = np.meshgrid([d2cq(i) for i in range(NQ)], [d2cv(i) for i in range(NV)])
    plt.pcolormesh(Q, DQ, pi.reshape((NV, NQ)), cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.title('Policy')
    plt.xlabel("q")
    plt.ylabel("dq")
    plt.show()
