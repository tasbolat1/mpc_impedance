import do_mpc
import numpy as np
np.random.seed(10)
from casadi import *
import time
import matplotlib.pyplot as plt

class CM_MPC():
    '''
    Generation circular motion centered around [x_c, y_c]
    '''
    def __init__(self, x_c, y_c, delta_t, target_angular_vel, radius):

        self.x_c = x_c
        self.y_c = y_c
        self.delta_t = delta_t
        self.ref = target_angular_vel

        # CONSTANTS
        self.m = 2 # dimension size: only x,y
        _k = 200 # stiffness: same as robot's 
        _d = 2*np.sqrt(_k) # damping: same as robots

        # PRECOMPUTED
        self.K = _k*np.eye(self.m)
        self.D = _d*np.eye(self.m)
        self.D_inv = np.linalg.inv(self.D)
        self.zeros_m_m = np.zeros(self.m)
        self.eye_m_m = np.eye(self.m)

        # Radius
        self.radius = radius

        # Initialize
        self.restart()


    def graphics_setup(self):
        self.mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)

    def model_setup(self):
        model_type = 'continuous'
        self.model = do_mpc.model.Model(model_type)

        # STATE
        self.pose = self.model.set_variable(var_type='_x', var_name='pose', shape=(self.m,1))
        self.theta_des = self.model.set_variable(var_type='_x', var_name='theta_des', shape=(1,1))
        self.theta_des_dot = self.model.set_variable(var_type='_x', var_name='theta_des_dot', shape=(1,1))
        self.state_size = 4

        # CONTROL
        self.theta_des_ddot = self.model.set_variable(var_type='_u', var_name='theta_des_ddot', shape=(1,1))

        # RHS
        pose_des_expr = vertcat(self.x_c + self.radius*cos(self.theta_des), self.y_c + self.radius*sin(self.theta_des))
        pose_dot_expr = self.D_inv@self.K@(pose_des_expr-self.pose)
        self.model.set_rhs("pose", pose_dot_expr)
        self.model.set_rhs("theta_des", self.theta_des_dot)
        self.model.set_rhs("theta_des_dot", self.theta_des_ddot)
        self.model.set_expression("pose_des", pose_des_expr)
        self.model.set_expression("pose_dot", pose_dot_expr)
        self.model.setup()

    def sim_setup(self):
        self.simulator = do_mpc.simulator.Simulator(self.model)
        self.simulator.set_param(t_step = self.delta_t)
        self.simulator.setup()

    def mpc_setup(self):
        self.mpc = do_mpc.controller.MPC(self.model)

        surpress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0, 'ipopt.max_wall_time':0.1}
        setup_mpc = {
            'n_horizon': 10,
            't_step': self.delta_t,
            'n_robust': 1,
            'store_full_solution': True,
            'nlpsol_opts':surpress_ipopt,
        }
        self.mpc.set_param(**setup_mpc)

        mterm = (self.ref-self.theta_des_dot)**2
        lterm = (self.ref-self.theta_des_dot)**2

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            theta_des_ddot=0#1e-2
        )

        # Bounds on states:
        self.mpc.bounds['lower','_x', 'pose', 0] = 0 # x
        self.mpc.bounds['upper','_x', 'pose', 0] = 1 # y

        self.mpc.bounds['lower','_x', 'pose', 1] = -0.5 # x
        self.mpc.bounds['upper','_x', 'pose', 1] = 0.5 # y

        # self.mpc.bounds['lower','_x', 'theta_des'] = -np.pi # x
        # self.mpc.bounds['upper','_x', 'theta_des'] = np.pi # y
        
        # Bounds on control
        self.mpc.bounds['lower','_u', 'theta_des_ddot'] = -5.5
        self.mpc.bounds['upper','_u', 'theta_des_ddot'] = 5.5

        self.mpc.setup()

    def set_x0(self, x0):
        self.simulator.x0 = x0
        self.mpc.x0 = x0

        self.mpc.set_initial_guess()

    def restart(self):
        # Initialize
        print('Restarting.')
        self.model_setup()
        self.sim_setup()
        self.mpc_setup()
        self.graphics_setup()

 

if __name__ == '__main__':

    x_c = 4.56838317e-01
    y_c = -2.40397883e-04

    mpc_controller = CM_MPC(x_c=x_c,
                            y_c=y_c,
                            delta_t=0.1,
                            target_angular_vel=1.0,
                            radius=0.1)

    pose_x0 = np.array([0.5571318446562233, -0.0009131756511991303]).reshape(-1,1)
    theta_des = np.array([0]).reshape(-1,1)
    theta_des_dot = np.array([0]).reshape(-1,1)

    x0 =  np.concatenate([pose_x0, theta_des, theta_des_dot], axis=0)

    mpc_controller.set_x0(x0)

    for t in range(150):
        start_time = time.time()
        u0 = mpc_controller.mpc.make_step(x0)
        print('{}: Time for mpc solver:{}'.format(t, time.time()-start_time))
        start_time = time.time()

        if mpc_controller.mpc.data['success'][-1] == 1:
            x0 = mpc_controller.simulator.make_step(u0)
            print('{}: Time for simulation:{}'.format(t, time.time()-start_time))
        else:
            print("Failed!")
            print("Stopping Controller ...")
            break

    #     print(mpc_controller.mpc.data.export().keys())
    #     break
    
    # sss = mpc_controller.mpc.data.prediction(('_x', 'theta_des_dot'), t_ind=-1)
    # print(sss)
        


    # fig, ax = plt.subplots(nrows=4, figsize=(16,9))
    # mpc_controller.mpc_graphics.add_line(var_type='_x', var_name='theta_des_dot', axis=ax[0])
    # mpc_controller.mpc_graphics.plot_predictions()
    # plt.show()
    fig, ax = plt.subplots(nrows=4, figsize=(16,9))
    ax[0].plot(mpc_controller.mpc.data['_time'], mpc_controller.mpc.data['_x'][:,3], label='theta_dot')
    ax[0].legend()
    ax[1].plot(mpc_controller.mpc.data['_time'], mpc_controller.mpc.data['_aux'][:,1:3], label='pose_des', linestyle='--')
    ax[1].legend()
    ax[2].plot(mpc_controller.mpc.data['_time'], mpc_controller.mpc.data['_x'][:,2], label='theta_des')
    ax[2].legend()
    ax[3].plot(mpc_controller.mpc.data['_time'], mpc_controller.mpc.data['_x'][:,0:2], label='pose')
    ax[3].legend()
    plt.show()

    fig, ax = plt.subplots(nrows=1, figsize=(16,9))
    ax.plot(mpc_controller.mpc.data['_x'][:,0], mpc_controller.mpc.data['_x'][:,1], label='theta_dot')
    plt.show()




