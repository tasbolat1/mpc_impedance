import rospy
from mpc import CM_MPC
from std_msgs.msg import Float32MultiArray, String
import numpy as np

class MPCController(object):
    def __init__(self):
        super(MPCController, self).__init__()
        rospy.init_node('mpc_node', anonymous=False)

        # define controller
        x_c = 0.4
        y_c = 0.0
        self.controller = CM_MPC(x_c=x_c,
                                y_c=y_c,
                                delta_t=0.2,
                                target_angular_vel=0.2)

        rospy.Subscriber('mpc/init_x0', Float32MultiArray, self.init_x0_Callback, queue_size=1)
        rospy.Subscriber('mpc/states', Float32MultiArray, self.state_Callback, queue_size=1)
        rospy.Subscriber('mpc/restart', String, self.restart_Callback, queue_size=1)
        self.cmd_pub = rospy.Publisher('mpc/controls', Float32MultiArray, queue_size=1)

        self.initialized = False

    def restart_Callback(self, msg):
        if msg.data == 'restart':
            self.controller.restart()
            self.initialized = False        

    def init_x0_Callback(self, msg):
        input_array = np.array(msg.data)
        if len(input_array) != self.controller.state_size:
            print('Wrong number of initial states assigned. Nothing set!')
            print('Required: {}, assigned: {}'.format(self.controller.state_size, len(input_array)))
        else:
            self.controller.set_x0(input_array.reshape(-1,1))
            self.initialized = True
            print('Initial states are set')

    def state_Callback(self, msg):

        if not self.initialized:
            print('The controller is not initialized yet! Set x0 values.')
            return

        input_array = np.array(msg.data)
        if len(input_array) != self.controller.state_size:
            print('Wrong number of initial states assigned. Nothing set!')
            print('Required: {}, assigned {}'.format(self.controller.state_size, len(input_array)))
        else:
            u0 = self.controller.mpc.make_step(input_array.reshape(-1,1))[0]

            if self.controller.mpc.data['success'] == 1: 
                control_array = Float32MultiArray()
                control_array.data = u0.tolist()
                self.cmd_pub.publish(control_array)
            else:
                print('Controller is failed!!!')

if __name__ == '__main__':
    mpc_controller = MPCController()
    rospy.spin()
