import rospy
import sys
import copy
import math
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler

# Initialize MoveIt! commander
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('moving_irb120_robot', anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
arm_group = moveit_commander.MoveGroupCommander("manipulator")

# Set planning parameters
arm_group.set_planning_time(10)  # Increase planning time
arm_group.set_num_planning_attempts(10)  # Increase number of attempts

# Publishers
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)
joint_state_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
waypoints_publisher = rospy.Publisher('/waypoints', PoseArray, queue_size=10)

# PID Controller
class PIDController:
    def __init__(self, p_gain, i_gain, d_gain):
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return (self.p_gain * error) + (self.i_gain * self.integral) + (self.d_gain * derivative)

# Initialize PID controllers for each joint (example values)
pid_controllers = [
    PIDController(1.0, 0.01, 0.1),  # Joint 1
    PIDController(1.0, 0.01, 0.1),  # Joint 2
    PIDController(1.0, 0.01, 0.1),  # Joint 3
    PIDController(1.0, 0.01, 0.1),  # Joint 4
    PIDController(1.0, 0.01, 0.1),  # Joint 5
    PIDController(1.0, 0.01, 0.1)   # Joint 6
]

def compute_cartesian_path(waypoints, eef_step, jump_threshold):
    (plan, fraction) = arm_group.compute_cartesian_path(
        waypoints,   # waypoints to follow
        eef_step,    # eef_step in meters
        jump_threshold)  # jump_threshold
    rospy.loginfo(f"Computed path fraction: {fraction}")
    return plan, fraction

def move_pose_arm(roll, pitch, yaw, x, y, z):
    pose_goal = Pose()
    quat = quaternion_from_euler(roll, pitch, yaw)
    pose_goal.orientation.x = quat[0]
    pose_goal.orientation.y = quat[1]
    pose_goal.orientation.z = quat[2]
    pose_goal.orientation.w = quat[3]
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    
    # Attempt to plan and execute motion
    for attempt in range(5):  # Try up to 5 times
        arm_group.set_pose_target(pose_goal)
        plan = arm_group.go(wait=True)
        if plan:
            break  # If plan is successful, break out of loop
        rospy.logwarn(f"Planning attempt {attempt + 1} failed")

    arm_group.stop()  # To guarantee no residual movement
    arm_group.clear_pose_targets()


def generate_circular_waypoints(center, radius, num_waypoints):
    waypoints = []
    for i in range(num_waypoints):
        angle = 2 * math.pi * i / num_waypoints
        wpose = arm_group.get_current_pose().pose
        wpose.position.x = center[0] + radius * math.cos(angle)
        wpose.position.y = center[1] + radius * math.sin(angle)
        wpose.position.z = center[2]
        waypoints.append(copy.deepcopy(wpose))
    return waypoints

def move_along_waypoints():
    center = [0.5, 0.0, 0.4]  # Center of the circle in XY plane
    radius = 0.1  # Radius of the circle
    num_waypoints = 100  # Number of waypoints

    waypoints = generate_circular_waypoints(center, radius, num_waypoints)

    # Log waypoints for debugging
    for i, waypoint in enumerate(waypoints):
        rospy.loginfo(f"Waypoint {i}: {waypoint}")

    # Publish waypoints for visualization
    pose_array = PoseArray()
    pose_array.header = Header()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.poses = waypoints
    waypoints_publisher.publish(pose_array)

    # Compute the Cartesian path
    plan, fraction = compute_cartesian_path(waypoints, 0.01, 0.0)

    # Check if the plan is valid
    if fraction > 0.9:
        rospy.loginfo("Path computed successfully. Moving the arm.")
        arm_group.execute(plan, wait=True)
    else:
        rospy.logwarn("Path planning failed.")

def publish_joint_states():
    joint_state_msg = JointState()
    joint_state_msg.header = Header()
    joint_state_msg.header.stamp = rospy.Time.now()  # Ensure the timestamp is current
    joint_state_msg.name = arm_group.get_active_joints()
    joint_state_msg.position = arm_group.get_current_joint_values()
    joint_state_publisher.publish(joint_state_msg)

if __name__ == '__main__':
    try:
        rospy.loginfo("Moving arm to HOME point")
        move_pose_arm(0, 0.8, 0, 0.4, 0, 0.6)
        rospy.sleep(1)

        # Move along waypoints
        rospy.loginfo("Moving along waypoints")
        move_along_waypoints()
        rospy.sleep(1)

       
        rospy.loginfo("All movements finished. Shutting down")
        moveit_commander.roscpp_shutdown()
    except rospy.ROSInterruptException:
        pass
