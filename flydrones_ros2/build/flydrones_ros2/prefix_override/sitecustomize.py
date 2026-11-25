import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/matheo/Desktop/crazyflie/flydrones_ros2/install/flydrones_ros2'
