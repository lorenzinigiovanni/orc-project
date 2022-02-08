from pendulum import Pendulum
import numpy as np
from numpy import pi
import time


# Readapted from "dpendulum.py"
class MyPendulum:
    ''' 
    Mix of discrete and continuos Pendulum environment.
    Joint angle and velocity are continuos.
    Torque are discretized with the specified steps.
    Torque is saturated.
    Guassian noise can be added in the dynamics. 
    '''

    def __init__(self, n_joints, nu=11, uMax=2, dt=0.1, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(n_joints, noise_stddev)
        self.pendulum.DT = dt
        self.pendulum.NDT = ndt
        # Number of discretization steps for joint torque
        self.nu = nu            
        # Max torque (u in [-umax,umax])
        self.uMax = uMax
        # Time step
        self.dt = dt
        # Discretization resolution for joint torque
        self.DU = 2*uMax/nu 

    @property
    def nx(self):
        return self.pendulum.nx

    # Discrete to continuous torque
    def d2cu(self, iu):
        iu = np.clip(iu, 0, self.nu-1) - (self.nu-1)/2
        return iu*self.DU

    # Put the robot in initial state
    def reset(self, x=None):
        self.x = self.pendulum.reset(x).flatten()
        return self.x

    # Perform a movement given a control
    def step(self, iu):
        self.x, cost = self.dynamics(self.x, iu)
        return self.x, cost

    # Render robot movement
    def render(self):
        q = self.x[:self.pendulum.nq]
        self.pendulum.display(q)
        time.sleep(self.pendulum.DT)

    # Call pendulum dynamics, with control u discretized
    def dynamics(self, x, iu):
        u = self.d2cu(iu)
        self.xc, cost = self.pendulum.dynamics(x, u)
        return self.xc, cost

if __name__ == "__main__":
    print("Start tests")
    env = MyPendulum()
    print("Tests finished")
