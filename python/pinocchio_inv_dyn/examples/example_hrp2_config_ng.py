from math import sqrt
import numpy as np

''' *********************** USER-PARAMETERS *********************** '''
SOLVER_CLASSIC       = 0;    # classic TSID formulation
SOLVER_ROBUST        = 1;
SOLVER_TO_INTEGRATE         = [SOLVER_CLASSIC,SOLVER_ROBUST];
ADD_ERRORS                  = True;
DATA_FILE_NAME              = 'data';
TEXT_FILE_NAME              = 'results.txt';
SAVE_DATA                   = True;

''' INITIAL STATE PARAMETERS '''
#MAX_TEST_DURATION           = 6000;
MAX_TEST_DURATION           = 5000;
dt                          = 1e-3;
model_path                  = ["/opt/openrobots/share"];
urdfFileName                = model_path[0] + "/hrp2_14_description/urdf/hrp2_14_reduced.urdf";
freeFlyer                   = True;
q0 = np.matrix([0.0, 0.0, 0.648702, 0.0, 0.0 , 0.0, 1.0,                             # Free flyer 0-6
                0.0, 0.0, 0.0, 0.0,                                                  # CHEST HEAD 7-10
                0.261799388,  0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # LARM       11-17
                0.261799388, -0.174532925, 0.0, -0.523598776, 0.0, 0.0, 0.174532925, # RARM       18-24
                0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # LLEG       25-30
                0.0, 0.0, -0.453785606, 0.872664626, -0.41887902, 0.0,               # RLEG       31-36
                ]).T;
                
q_hslff = np.matrix([ 0.  ,  0.  ,  0.62,  0.  ,  0.  ,  0. ,1 ,  0.  ,  0.  ,
        0.  ,  0.  ,  0.26,  0.17,  0.  , -0.52,  0.  ,  0.  ,  0.1 ,
        0.26, -0.17,  0.  , -0.52,  0.  ,  0.  ,  0.1 ,  0.  ,  0.  ,
       -0.75,  0.87, -0.12,  0.  ,  0.  ,  0.  , -0.15,  0.87, -0.72,  0. ]).T;

v0 = np.matrix(np.zeros(36)).T;

''' CONTROLLER CONFIGURATION '''
ENABLE_CAPTURE_POINT_LIMITS     = True;
ENABLE_TORQUE_LIMITS            = True;
ENABLE_FORCE_LIMITS             = True;
ENABLE_JOINT_LIMITS             = True;
IMPOSE_POSITION_BOUNDS          = True;
IMPOSE_VELOCITY_BOUNDS          = True;
IMPOSE_VIABILITY_BOUNDS         = True;
IMPOSE_ACCELERATION_BOUNDS      = True;
JOINT_POS_PREVIEW               = 1.5; # preview window to convert joint pos limits into joint acc limits
JOINT_VEL_PREVIEW               = 1;   # preview window to convert joint vel limits into joint acc limits
#MAX_JOINT_ACC                   = 30.0;
MAX_JOINT_ACC                   = 10.0;
MAX_MIN_JOINT_ACC               = 10.0;
USE_JOINT_VELOCITY_ESTIMATOR    = False;
ACCOUNT_FOR_ROTOR_INERTIAS      = True;
# rh task
TRAJECTORY_TIME_REACHABLE                 = 1.0;
TRAJECTORY_TIME_UNREACHABLE               = 3.0;
REACHABLE                       = 0;
UNREACHABLE                     = 1;
RH_TASK_SWITCH                  = REACHABLE;
# CONTROLLER GAINS
kp_posture  = 100.0; #1.0;   # proportional gain of postural task
kd_posture  = 2*sqrt(kp_posture);
kp_constr   = 100;   # constraint proportional feedback gain
kd_constr   = 2*sqrt(kp_constr);   # constraint derivative feedback gain
kp_com      = 10.0;
kd_com      = 2*sqrt(kp_com);
kp_rh_unreachable       = 5;
kd_rh_unreachable       = 2*sqrt(kp_rh_unreachable);
kp_rh_reachable         = 100;
kd_rh_reachable         = 2*sqrt(kp_rh_reachable);


import pinocchio as se3
# unreachable
tr_ur = np.matrix((1.02,-0.3,0.9)).T
rot = np.matrix(((0.,0.,-1.0),(0.,1.,0.),(1.,0.,0.)))
rh_des_unreachable = se3.SE3.Random()
rh_des_unreachable.translation = tr_ur
rh_des_unreachable.rotation = rot
#reachable
tr_r = np.matrix((0.8,-0.27,0.9)).T
rh_des_reachable   = se3.SE3.Random()
rh_des_reachable.translation = tr_r
rh_des_reachable.rotation = rot

      
constraint_mask = np.array([True, True, True, True, True, True]).T;
rh_mask         = np.array([True, True, True, False, False, False]).T;
# CONTROLLER WEIGTHS
w_com           = 1;
w_posture       = 1e-2;  # weight of postural task
w_rh            = 1;



# QP SOLVER PARAMETERS
maxIter = 300;      # max number of iterations
maxTime = 0.8;      # max computation time for the solver in seconds
verb=0;             # verbosity level (0, 1, or 2)

# CONTACT PARAMETERS
mu  = np.array([0.3, 0.1]);          # force and moment friction coefficient
fMin = 1e-3;					     # minimum normal force

''' SIMULATOR PARAMETERS '''
FORCE_TORQUE_LIMITS            = ENABLE_TORQUE_LIMITS;
FORCE_JOINT_LIMITS             = ENABLE_JOINT_LIMITS and IMPOSE_POSITION_BOUNDS;
USE_LCP_SOLVER                 = False

''' STOPPING CRITERIA THRESHOLDS '''
MAX_CONSTRAINT_ERROR        = 0.1;

'''INERTIAL ERROR'''
MAX_COM_ERROR = 0.02
MAX_MASS_ERROR = 0.1
MAX_INERTIA_ERROR = 0.01

''' INITIAL STATE PARAMETERS '''
INITIAL_CONFIG_ID                   = 0;
INITIAL_CONFIG_FILENAME             = '../../../data/hrp2_configs_coplanar';

''' VIEWER PARAMETERS '''
ENABLE_VIEWER               = False;
PLAY_MOTION_WHILE_COMPUTING = True;
PLAY_MOTION_AT_THE_END      = True;
DT_VIEWER                   = 10*dt;   # timestep used to display motion with viewer
SHOW_VIEWER_FLOOR           = True;

''' FIGURE PARAMETERS '''
SAVE_FIGURES     = False;
SHOW_FIGURES     = False;
SHOW_LEGENDS     = True;
LINE_ALPHA       = 0.7;
#BUTTON_PRESS_TIMEOUT        = 100.0;

GEAR_RATIO = (  0.  ,    0.  ,    0.  ,    0.  ,    0.  ,    0.  ,    1.  ,
          207.69,  381.54,  100.  ,  100.  ,  219.23,  231.25,  266.67,
          250.  ,  145.45,  350.  ,  200.  ,  219.23,  231.25,  266.67,
          250.  ,  145.45,  350.  ,  200.  ,  384.  ,  240.  ,  180.  ,
          200.  ,  180.  ,  100.  ,  384.  ,  240.  ,  180.  ,  200.  ,
          180.  ,  100.  )
              
INERTIA_ROTOR = (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
           0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
           6.96000000e-06,   6.96000000e-06,
           1.10000000e-06,   1.10000000e-06,   6.96000000e-06,
           6.60000000e-06,   1.00000000e-06,   6.60000000e-06,
           1.10000000e-06,   1.00000000e-06,   1.00000000e-06,
           6.96000000e-06,   6.60000000e-06,   1.00000000e-06,
           6.60000000e-06,   1.10000000e-06,   1.00000000e-06,
           1.00000000e-06,   1.01000000e-06,   6.96000000e-06,
           1.34000000e-06,   1.34000000e-06,   6.96000000e-06,
           6.96000000e-06,   1.01000000e-06,   6.96000000e-06,
           1.34000000e-06,   1.34000000e-06,   6.96000000e-06,
           6.96000000e-06);