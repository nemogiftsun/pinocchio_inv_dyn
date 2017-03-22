'''
Simple example of how to use this simulation/control environment using HRP-2.
'''
import example_hrp2_config_ng as conf
import pinocchio as se3
from pinocchio_inv_dyn.polytope_conversion_utils import poly_face_to_span
from pinocchio.utils import *
from pinocchio.utils import zero as mat_zeros
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
from pinocchio_inv_dyn.standard_qp_solver import StandardQpSolver
from pinocchio_inv_dyn.simulator import Simulator
import pinocchio_inv_dyn.viewer_utils as viewer_utils
from pinocchio_inv_dyn.inv_dyn_formulation_util import InvDynFormulation
from pinocchio_inv_dyn.tasks import SE3Task, CoMTask, JointPostureTask
from pinocchio_inv_dyn.trajectories import SmoothedSE3Trajectory, ConstantSE3Trajectory, ConstantNdTrajectory, MinJerkSE3Trajectory
import pinocchio_inv_dyn.plot_utils as plot_utils
from pinocchio_inv_dyn.add_inertial_uncertainties import generate_new_inertial_params
from pinocchio_inv_dyn.data_structure_utils import Bunch
import cProfile
import pickle
import numpy as np
import os
from numpy.linalg import norm
from datetime import datetime
from time import sleep

EPS = 1e-4;

def createListOfMatrices(listSize, matrixSize):
    l = listSize*[None,];
    for i in range(listSize):
        l[i] = np.matlib.zeros(matrixSize);
    return l;

def createListOfLists(size1, size2):
    l = size1*[None,];
    for i in range(size1):
        l[i] = size2*[None,];
    return l;
    
def createInvDynFormUtil(q, v,invdyn_configs,id_solver):
    invDynForm = InvDynFormulation('inv_dyn'+datetime.now().strftime('%m%d_%H%M%S'), 
                                   q, v, invdyn_configs);           
    invDynForm.enableTorqueLimits(conf.ENABLE_TORQUE_LIMITS);
    invDynForm.enableForceLimits(conf.ENABLE_FORCE_LIMITS);
    invDynForm.enableJointLimits(conf.ENABLE_JOINT_LIMITS, conf.IMPOSE_POSITION_BOUNDS, conf.IMPOSE_VELOCITY_BOUNDS, 
                                 conf.IMPOSE_VIABILITY_BOUNDS, conf.IMPOSE_ACCELERATION_BOUNDS);
    invDynForm.JOINT_POS_PREVIEW            = conf.JOINT_POS_PREVIEW;
    invDynForm.JOINT_VEL_PREVIEW            = conf.JOINT_VEL_PREVIEW;
    invDynForm.MAX_JOINT_ACC                = conf.MAX_JOINT_ACC;
    invDynForm.MAX_MIN_JOINT_ACC            = conf.MAX_MIN_JOINT_ACC;
    invDynForm.USE_JOINT_VELOCITY_ESTIMATOR = conf.USE_JOINT_VELOCITY_ESTIMATOR;
    invDynForm.ACCOUNT_FOR_ROTOR_INERTIAS   = conf.ACCOUNT_FOR_ROTOR_INERTIAS;  
    return invDynForm;

def createSimulator(q0, v0,simulator_configs):
    simulator  = Simulator('hrp2_sim'+datetime.now().strftime('%Y%m%d_%H%M%S')+str(np.random.random()),q0,v0,simulator_configs);
    simulator.viewer.CAMERA_FOLLOW_ROBOT = False;
    simulator.USE_LCP_SOLVER = conf.USE_LCP_SOLVER;
    simulator.ENABLE_TORQUE_LIMITS = conf.FORCE_TORQUE_LIMITS;
    simulator.ENABLE_FORCE_LIMITS = conf.ENABLE_FORCE_LIMITS;
    simulator.ENABLE_JOINT_LIMITS = conf.FORCE_JOINT_LIMITS;
    simulator.ACCOUNT_FOR_ROTOR_INERTIAS = conf.ACCOUNT_FOR_ROTOR_INERTIAS;
    simulator.VIEWER_DT = conf.DT_VIEWER;
    simulator.CONTACT_FORCE_ARROW_SCALE = 2e-3;
    simulator.verb=0;
    return simulator;
    
def updateConstraints(t, q, v, invDynForm, contacts):
    contact_changed = False;
    
    for (name, PN) in contacts.iteritems():
        if(invDynForm.existUnilateralContactConstraint(name)):
            continue;
        
        contact_changed =True;
        invDynForm.r.forwardKinematics(q, v, 0 * v);
        invDynForm.r.framesKinematics(q);
        
        fid = invDynForm.getFrameId(name);
        oMi = invDynForm.r.framePosition(fid);
        ref_traj = ConstantSE3Trajectory(name, oMi);
        constr = SE3Task(invDynForm.r, fid, ref_traj, name);
        constr.kp = conf.kp_constr;
        constr.kv = conf.kd_constr;
        constr.mask(conf.constraint_mask);
        
        Pi = np.matrix(PN['P']).T;
        Ni = np.matrix(PN['N']).T;
        for j in range(Pi.shape[1]):
            print "    contact point %d in world frame:"%j, oMi.act(Pi[:,j]).T, (oMi.rotation * Ni[:,j]).T;
        invDynForm.addUnilateralContactConstraint(constr, Pi, Ni, conf.fMin, conf.mu);
        simulator.addUnilateralContactConstraint(constr, Pi, Ni, conf.mu);

    return contact_changed;
    
            
def startSimulation(q0, v0, j,safety_margin):
    FIRST_TARGET_REACHED=False
    solverId = conf.SOLVER_TO_INTEGRATE[j]
    print '\nGONNA INTEGRATE CONTROLLER %d' % solverId;
    capturePointIn = True;
    
    constrViol          = np.empty(conf.MAX_TEST_DURATION).tolist(); #list of lists of lists
    constrViolString    = '';    
    torques = np.zeros(na);

    t = 0;
    # CHECK CHECK CHECK
    
    invDynForm.POLYTOPE_MARGIN = safety_margin;
    if (solverId == conf.SOLVER_CLASSIC_WITH_AVG) or (solverId == conf.SOLVER_CLASSIC_WITH_MIN) or (solverId == conf.SOLVER_CLASSIC_WITH_REG) or (solverId == conf.SOLVER_CLASSIC_WITH_MAX):   
        invDynForm.setNewSensorData(0, q0, v0);                                 
        invDynForm.enableCapturePointLimits(conf.ENABLE_CAPTURE_POINT_LIMITS);
    elif solverId == conf.SOLVER_ROBUST:
        invDynForm.setNewSensorData(0, q0, v0);
        invDynForm.enableCapturePointLimitsRobust(conf.ENABLE_CAPTURE_POINT_LIMITS,False);
    elif solverId == conf.SOLVER_ROBUST_VEL:
        invDynForm.setNewSensorData(0, q0, v0);
        invDynForm.enableCapturePointLimitsRobust(conf.ENABLE_CAPTURE_POINT_LIMITS,True);        
                
    simulator.reset(t, q0, v0, conf.dt);
    print 'The Polytope Safety Margin is '
    print invDynForm.POLYTOPE_MARGIN
    
    contact_names  = [con.name for con in invDynForm.rigidContactConstraints];
    contact_sizes  = [con.dim for con in invDynForm.rigidContactConstraints];
    contact_size_cum = [int(np.sum(contact_sizes[:ii])) for ii in range(len(contact_sizes))];
    contact_points = [con.framePosition().translation for con in invDynForm.rigidContactConstraints];

    for i in range(conf.MAX_TEST_DURATION):   
                                
        invDynForm.setNewSensorData(t, simulator.q, simulator.v);
        (G,glb,gub,lb,ub) = invDynForm.createInequalityConstraints();        
        (D,d)             = invDynForm.computeCostFunction(t);

        q[j][:,i]         = np.matrix.copy(invDynForm.q);
        v[j][:,i]         = np.matrix.copy(invDynForm.v);
        x_com[j][:,i]     = np.matrix.copy(invDynForm.x_com);       # from the solver view-point
        dx_com[j][:,i]    = np.matrix.copy(invDynForm.dx_com);      # from the solver view-point
        cp[j][:,i]        = np.matrix.copy(invDynForm.cp);          # from the solver view-point
        ang_mom[j,i]      = norm(invDynForm.getAngularMomentum());
        
        rh_current_tr = simulator.r.framePosition(simulator.r.model.getFrameId('RARM_JOINT5')).translation
        rh_task_error[j,i] = np.linalg.norm(rh_current_tr - rh_des_tr);
        if (FIRST_TARGET_REACHED==False) and (i!=0): 
                if(( rh_task_error[j,i] < 5e-3 or conf.RH_TASK_SWITCH == conf.UNREACHABLE) and (np.max(np.abs(v[j][:,i]))<0.001)):
                    print "\n\n First target reached in %.3f s!"%((t)),"Right hand task error",rh_task_error[j,i];
                    FIRST_TARGET_REACHED = True;
                    time_to_reach[j] = t;
                    i_to_reach[j]    = i;
                    sleep(1);
                    final_time[j] = t;
                    final_time_step[j] = i;
                    return False;   
        
        if(i%500==0):
            print "Time %.3f... i %d" % (t, i), "Max joint vel", np.max(np.abs(v[j][:,i])),"Right hand task error",rh_task_error[j,i];
        
        if(i==conf.MAX_TEST_DURATION-1):
            print "MAX TIME REACHED \n";
            print "Max joint vel", np.max(np.abs(v[j][:,i]));
            final_time[j]       = t;
            final_time_step[j]  = i;
            return True;

        ''' tell the solvers that if the QP is unfeasible they can relax the joint-acc inequality constraints '''        
        solvers[j].setSoftInequalityIndexes(invDynForm.ind_acc_in);
        m_in = glb.size;
        solvers[j].changeInequalityNumber(m_in);
        (torques, solver_imode[j,i])    = solvers[j].solve(D.A, d.A, G.A, glb.A, gub.A, lb.A, ub.A, torques, maxTime=conf.maxTime);

        tau[j][:,i]                 = np.matrix.copy(torques).reshape((na,1));
        y                           = invDynForm.C * tau[j][:,i] + invDynForm.c;
        dv[j][:,i]                  = y[:nv];
        (tmp1, tmp2, ddx_com[j][:,i]) = invDynForm.r.com(q[j][:,i], v[j][:,i], dv[j][:,i]); #J_com * dv[j][:,i] + invDynForm.dJcom_dq;
        n_active_ineq[j,i]          = solvers[j].nActiveInequalities;   # from the solver view-point
        n_violated_ineq[j,i]        = solvers[j].nViolatedInequalities; # from the solver view-point                    
#        ineq[i,j,:m_in]             = np.dot(G, tau[i,j,:]) - glb; # from the solver view-point

        if(np.isnan(torques).any() or np.isinf(torques).any()):
            no_sol_count[j] += 1;

        #simulator.computeForwardDynamicMapping(t);

        constrViol[i] = simulator.integrate(t, dt, tau[j][:,i])
#        constrViol[i] = simulator.integrateAcc(t, dt, dv[j][:,i], conf.PLAY_MOTION_WHILE_COMPUTING);        
        # update viewer
        #simulator.updateComPositionInViewer(np.matrix([x_com[j][0,i], x_com[j][1,i], 0.]).T);
        cp_real[j][:,i]   = np.matrix.copy(simulator.cp);
        simulator.updateCapturePointPositionInViewer(cp[j][:,i],cp_real[j][:,i]);#cp_real
        if solverId == conf.SOLVER_ROBUST:
            v_cp_robust[j][i] = invDynForm.vdcom
        elif solverId == conf.SOLVER_ROBUST_VEL:
            v_cp_robust[j][i] = invDynForm.vdcom
            v_cp_robust_vel[j][i] = invDynForm.vdcom1            
        f              = y[nv:nv+invDynForm.k];
        contact_forces = [ f[ii:ii+3] for ii in contact_size_cum];
        simulator.updateContactForcesInViewer(contact_names, contact_points, contact_forces);
        
        for cv in constrViol[i]:
            cv.time = t;
            print cv.toString();
            constrViolString += cv.toString()+'\n';
            
        ''' CHECK TERMINATION CONDITIONS '''
        constraint_errors = [con.positionError(t) for con in invDynForm.rigidContactConstraints];
        for err in constraint_errors:
            if(norm(err[:3]) > conf.MAX_CONSTRAINT_ERROR):
                print "ERROR Time %.3f constraint error:"%t, err[:3].T;
                time_to_fall[j] = t; 
                i_to_fall[j] = i;
                final_time[j] = t;
                final_time_step[j] = i;
                falls[j] = 1;
                return False;
                
        ddx_c = invDynForm.Jc * dv[j][:,i] + invDynForm.dJc_v;
        constr_viol = ddx_c - invDynForm.ddx_c_des;
        if(norm(constr_viol)>EPS):
            print "Time %.3f Constraint violation:"%(t), norm(constr_viol), ddx_c.T, "!=", invDynForm.ddx_c_des.T;
            print "Joint torques:", torques.T
            time_to_fall[j] = t; 
            i_to_fall[j] = i;
            final_time[j] = t;
            final_time_step[j] = i;
            falls[j] = 1;
            return False;
            
        # Check whether robot is falling
        if(np.sum(n_violated_ineq[j,:]) > 10):
            print "Com velocity", np.linalg.norm(dx_com[j][:,i]);
            print "Solver violated %d inequalities" % solvers[j].nViolatedInequalities; #, "max inequality violation", np.min(ineq[i,j,:m_in]);
            print "ROBOT FELL AFTER %.3f s\n" % (t);
            time_to_fall[j] = t; 
            i_to_fall[j] = i;
            final_time[j] = t;
            final_time_step[j] = i;
            falls[j] = 1;
            for index in range(i+1,conf.MAX_TEST_DURATION):
                q[j][:,index] = q[j][:,i];
            return False;
            

        ''' ******************************************** DEBUG PRINTS ******************************************** '''
        cp_ineq = np.dot(B_sp, cp_real[j][:,i]) + b_sp;
        if(capturePointIn and (cp_ineq<-EPS).any()):
            print "Time %.3f WARNING capture point is outside support polygon, margin %.3f" % (t, np.min(cp_ineq))
            capturePointIn = False;
        elif(not capturePointIn and (cp_ineq>=0.0).all()):
            print "Time %.3f WARNING capture point got inside support polygon, margin %.3f" % (t, np.min(cp_ineq))
            capturePointIn = True;
        elif(not capturePointIn):
            print "Time %.3f WARNING capture point still outside support polygon, margin %.3f" % (t, np.min(cp_ineq))

        t += dt;
        sp_points = np.array(invDynForm.contact_points_reduced[0:2,:])
        simulator.viewer.addPolytope('sp_controller',sp_points,robotName='robot1',color = [0,0.7,0.4,0.8])
        if (solverId == conf.SOLVER_ROBUST or solverId == conf.SOLVER_ROBUST_VEL):
            simulator.viewer.addPolytope('capture point polytope',invDynForm.vdcom,robotName='robot1',color = [1,0.8,1,1])        
            simulator.viewer.addPolytope('capture point polytope with velocity',invDynForm.vdcom1,robotName='robot1',color =[1,0,0,1])  
            
            #simulator.viewer.addPolytope('capture point polytope with velocity_1',invDynForm.vcom_v,robotName='robot1',color =[1,0,1,1])  
        if(i%100==0):
            simulator.viewer.addPolytope('sp_controller'+str(s),sp_points,robotName='robot1',color = [s,0.7,0.4,0.8])
#    
        #simulator.viewer.addSphere('t1', 0.04, invDynForm.r.framePosition(invDynForm.getFrameId('RARM_JOINT6')).translation, color=(0.0, 0.2, 0, 1));
        #simulator.viewer.addSphere('t2', 0.04, invDynForm.r.framePosition(invDynForm.getFrameId('RARM_JOINT5')).translation, color=(0.0, 0.0, 0.5, 1));


''' *********************** BEGINNING OF MAIN SCRIPT *********************** '''
seed = int(np.random.uniform(0, 1000))
#seed = 336
np.random.seed(seed)
import random
random.seed(seed);
print "Random seed", seed;

np.set_printoptions(precision=2, suppress=True);
date_time = datetime.now().strftime('%Y%m%d_%H%M%S');
viewer_utils.ENABLE_VIEWER  = conf.ENABLE_VIEWER
plot_utils.FIGURE_PATH      = '../results/test_unreachable/'+date_time+'/'; #'_'+str(conf.SOLVER_TO_INTEGRATE).replace(' ', '_')+'/';
plot_utils.SAVE_FIGURES     = conf.SAVE_FIGURES;
plot_utils.SHOW_FIGURES     = conf.SHOW_FIGURES;
plot_utils.SHOW_LEGENDS     = conf.SHOW_LEGENDS;
plot_utils.LINE_ALPHA       = conf.LINE_ALPHA;
SHOW_CONTACT_POINTS_IN_VIEWER = False;
TEXT_FILE_NAME              = 'results.txt';

''' CREATE CONTROLLER AND SIMULATOR '''
if(conf.freeFlyer):
    robot = RobotWrapper(conf.urdfFileName, conf.model_path, root_joint=se3.JointModelFreeFlyer());
else:
    robot = RobotWrapper(conf.urdfFileName, conf.model_path, None);
nq = robot.nq;
nv = robot.nv;
dt = conf.dt;
#q0 = se3.randomConfiguration(robot.model, robot.model.lowerPositionLimit, robot.model.upperPositionLimit);
#q0 = conf.q0;
q0 = conf.q_hslff;
v0 = conf.v0;

#create invdyn form util
invdyn_configs = Bunch(dt = None, mesh_dir = None, urdfFileName = None, freeFlyer=True,vcom = None,ncom=None,inertiaError=[0,0,0],GEAR_RATIO=conf.GEAR_RATIO,INERTIA_ROTOR=conf.INERTIA_ROTOR,ACCOUNT_FOR_ROTOR_INERTIAS=conf.ACCOUNT_FOR_ROTOR_INERTIAS)
invdyn_configs.dt = conf.dt
invdyn_configs.mesh_dir = conf.model_path
invdyn_configs.urdfFileName = conf.urdfFileName
invdyn_configs.freeFlyer = conf.freeFlyer
invdyn_configs.inertiaError=[conf.MAX_MASS_ERROR,conf.MAX_COM_ERROR,conf.MAX_INERTIA_ERROR]
simulator_configs = Bunch(dt=None, model_path=None, freeFlyer = False,urdfFileName=None, fMin=None, mu=None,detectContactPoint=False,r_modified=None,GEAR_RATIO=conf.GEAR_RATIO,INERTIA_ROTOR=conf.INERTIA_ROTOR,ACCOUNT_FOR_ROTOR_INERTIAS=conf.ACCOUNT_FOR_ROTOR_INERTIAS)
invdyn_configs.vcom,invdyn_configs.ncom,error_model = generate_new_inertial_params(conf.MAX_MASS_ERROR,conf.MAX_COM_ERROR,conf.MAX_INERTIA_ERROR)
if conf.ADD_ERRORS == True:
    simulator_configs.r_modified = error_model
else:
    simulator_configs.r_modified = None;
invDynForm = createInvDynFormUtil(q0, v0,invdyn_configs,conf.SOLVER_TO_INTEGRATE[0]);
# create sim
simulator_configs.fMin=conf.fMin;
simulator_configs.mu=conf.mu;
simulator_configs.dt=conf.dt;
simulator_configs.freeFlyer = conf.freeFlyer
simulator_configs.mesh_dir=conf.model_path;
simulator_configs.urdfFileName=conf.urdfFileName;
simulator = createSimulator(q0, v0,simulator_configs);

robot = invDynForm.r;
na = invDynForm.na;    # number of joints
simulator.viewer.setVisibility("floor", "ON" if conf.SHOW_VIEWER_FLOOR else "OFF");
    
f = open(conf.INITIAL_CONFIG_FILENAME, 'rb');
res = pickle.load(f);
f.close();
contacts = res[conf.INITIAL_CONFIG_ID]['contact_points'];
del res;

updateConstraints(0, q0, v0, invDynForm, contacts);
(G,glb,gub,lb,ub) = invDynForm.createInequalityConstraints();
m_in = glb.size;
k = invDynForm.k;    # number of constraints
mass = invDynForm.M[0,0];

''' CREATE POSTURAL TASK '''   
posture_traj = ConstantNdTrajectory("posture_traj", q0[7:,0]);
posture_task = JointPostureTask(invDynForm.r, posture_traj);
posture_task.kp = conf.kp_posture;
posture_task.kv = conf.kd_posture;
invDynForm.addTask(posture_task, conf.w_posture);

''' CREATE Right hand task ''' 
fid = invDynForm.getFrameId('RARM_JOINT5');
oMi = invDynForm.r.framePosition(fid);

#righthand_traj = ConstantSE3Trajectory('RARM_JOINT5', rh_des); 
kp_rh = 0;kv_rh=0;
if conf.RH_TASK_SWITCH == conf.UNREACHABLE:
    rh_des = conf.rh_des_unreachable
    kp_rh = conf.kp_rh_unreachable;
    kv_rh = conf.kd_rh_unreachable;
    trajectory_time = conf.TRAJECTORY_TIME_UNREACHABLE;
elif conf.RH_TASK_SWITCH == conf.REACHABLE:
    rh_des = conf.rh_des_reachable
    kp_rh = conf.kp_rh_reachable;
    kv_rh = conf.kd_rh_reachable;
    trajectory_time = conf.TRAJECTORY_TIME_REACHABLE

rh_des_tr = rh_des.translation
righthand_traj = MinJerkSE3Trajectory('RARM_JOINT5', oMi, rh_des, conf.dt, trajectory_time);
#rh_traj_back = MinJerkSE3Trajectory('RARM_JOINT5', rh_des, oMi, conf.dt, trajectory_time);
#N = rh_traj_forth._x_ref.shape[1];
#x_rh_ref = (2*N)*[None,];
#for i in range(N):
#    x_rh_ref[i] = se3.SE3.Identity();
#    x_rh_ref[i].translation = rh_traj_forth._x_ref[:,i]
#for i in range(N):
#    x_rh_ref[N+i] = se3.SE3.Identity();
#    x_rh_ref[N+i].translation = rh_traj_back._x_ref[:,i];
#righthand_traj = SmoothedSE3Trajectory('RARM_JOINT5', x_rh_ref, conf.dt, 21)

task_rh_hand= SE3Task(invDynForm.r, fid, righthand_traj,'RARM_JOINT5');
task_rh_hand.kp = kp_rh;
task_rh_hand.kv = kv_rh; 
invDynForm.addTask(task_rh_hand, conf.w_rh)
task_rh_hand.mask(conf.rh_mask); 

simulator.viewer.addSphere('target', 0.03, rh_des.translation, color=(1.0, 0, 0, 1));

''' CREATE COM TASK '''
com_des = invDynForm.x_com.copy();
#com_des[1] += COM_DISTANCE;
com_traj  = ConstantNdTrajectory("com_traj", com_des);
com_task = CoMTask(invDynForm.r, com_traj);
com_task.kp = conf.kp_com;
com_task.kv = conf.kd_com;
#invDynForm.addTask(com_task, conf.w_com);

''' Compute stats about initial state '''
(B_sp, b_sp) = invDynForm.getSupportPolygon();
x_com_0  =  invDynForm.x_com.copy();
com_ineq = np.dot(B_sp, x_com_0[:2]) + b_sp;
in_or_out = "outside" if (com_ineq<0.0).any() else "inside";
print "initial com pos "+str(x_com_0.T)+" is "+in_or_out+" support polygon, margin: %.3f"%(np.min(com_ineq));

if(SHOW_CONTACT_POINTS_IN_VIEWER):
    p = invDynForm.contact_points.copy();
    p[2,:] -= 0.005;
    for j in range(p.shape[1]):
        simulator.viewer.addSphere("contact_point"+str(j), 0.005, p[:,j], (0.,0.,0.), (1, 1, 1, 1));
        simulator.viewer.updateObjectConfigRpy("contact_point"+str(j), p[:,j]);
invDynForm.contact_points
sp_points = np.array(invDynForm.contact_points[0:2,:])
v_sp_original        = sp_points
v_sp_reduced         = np.array(invDynForm.contact_points_reduced[0:2,:])
ROBOT_MOVE = np.array((0.5,0.5,-0.01))

#simulator.viewer.addPolytope('sp_controller',(sp_points.T).T,robotName='robot1',color = [0,0.7,0.4,0.8])

''' Create the solvers '''
solver_id       = StandardQpSolver(na, m_in, "qpoases", maxIter=conf.maxIter, verb=conf.verb);
#if conf.SOLVER_CLASSIC in conf.SOLVER_TO_INTEGRATE:
#    solvers = [solver_id,] * (len(conf.SOLVER_TO_INTEGRATE) + len(conf.SAFETY_MARGIN) - 1)
#else:    
solvers = [solver_id,] * len(conf.SOLVER_TO_INTEGRATE);
N_SOLVERS = len(solvers);
solver_names = [s.name for s in solvers];

q                    = createListOfMatrices(N_SOLVERS, (nq, conf.MAX_TEST_DURATION));
v                    = createListOfMatrices(N_SOLVERS, (nv, conf.MAX_TEST_DURATION));
fc                   = createListOfMatrices(N_SOLVERS, (k, conf.MAX_TEST_DURATION));
tau                  = createListOfMatrices(N_SOLVERS, (na, conf.MAX_TEST_DURATION));
dv                   = createListOfMatrices(N_SOLVERS, (nv, conf.MAX_TEST_DURATION));
#ineq                 = createListOfMatrices(N_SOLVERS, (m_in, conf.MAX_TEST_DURATION));
x_com                = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
dx_com               = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
ddx_com              = createListOfMatrices(N_SOLVERS, (3, conf.MAX_TEST_DURATION));
cp                   = createListOfMatrices(N_SOLVERS, (2, conf.MAX_TEST_DURATION));   # capture point
cp_real              = createListOfMatrices(N_SOLVERS, (2, conf.MAX_TEST_DURATION));
zmp                  = createListOfMatrices(N_SOLVERS, (2, conf.MAX_TEST_DURATION));   # zmp
ang_mom              = np.zeros((N_SOLVERS, conf.MAX_TEST_DURATION));      # angular momentum
n_active_ineq        = np.zeros((N_SOLVERS, conf.MAX_TEST_DURATION), dtype=np.int);
n_violated_ineq      = np.zeros((N_SOLVERS, conf.MAX_TEST_DURATION), dtype=np.int);
no_sol_count         = np.zeros(N_SOLVERS, dtype=np.int);
solver_imode         = np.zeros((N_SOLVERS, conf.MAX_TEST_DURATION), dtype=np.int);
final_time           = np.zeros(N_SOLVERS);
final_time_step      = np.zeros(N_SOLVERS, np.int);
controller_balance   = N_SOLVERS*[False,];
# reaching task relavant 
time_to_fall         = np.zeros(N_SOLVERS);
time_to_reach        = np.zeros(N_SOLVERS);
i_to_fall            = np.zeros(N_SOLVERS);
i_to_reach           = np.zeros(N_SOLVERS);
rh_task_error        = np.zeros((N_SOLVERS,conf.MAX_TEST_DURATION));




v_cp_robust          = [[[None] for i in range(conf.MAX_TEST_DURATION)] for i in range(N_SOLVERS)]
v_cp_robust_vel      = [[[None] for i in range(conf.MAX_TEST_DURATION)] for i in range(N_SOLVERS)]
v_sp                 = [[None]*conf.MAX_TEST_DURATION]*N_SOLVERS
itmax                = np.ones(N_SOLVERS)*conf.MAX_TEST_DURATION
falls                = np.zeros(N_SOLVERS);
time_max             = np.zeros(N_SOLVERS);


if(conf.SAVE_DATA):
    os.makedirs(plot_utils.FIGURE_PATH);
    np.savez_compressed(plot_utils.FIGURE_PATH+conf.DATA_FILE_NAME+'_config', 
                        SOLVER_TO_INTEGRATE=conf.SOLVER_TO_INTEGRATE,
                        MAX_MASS_ERROR=conf.MAX_MASS_ERROR,
                        MAX_COM_ERROR=conf.MAX_COM_ERROR,
                        MAX_INERTIA_ERROR=conf.MAX_INERTIA_ERROR,
                        dt=conf.dt,
                        maxTime=conf.maxTime,
                        maxIter=conf.maxIter,
                        kp_posture=conf.kp_posture,
                        kd_posture=conf.kd_posture,
                        w_posture=conf.w_posture,
                        kp_rh=kp_rh,
                        kd_rh=kv_rh,
                        w_rh=conf.w_rh,
                        mu=conf.mu,
                        fMin=conf.fMin,
                        JOINT_POS_PREVIEW=conf.JOINT_POS_PREVIEW,
                        JOINT_VEL_PREVIEW=conf.JOINT_VEL_PREVIEW,
                        MAX_JOINT_ACC=conf.MAX_JOINT_ACC,
                        ACCOUNT_FOR_ROTOR_INERTIAS=conf.ACCOUNT_FOR_ROTOR_INERTIAS
                        );

#print simulator.r.model.inertias[simulator.r.model.getJointId('RARM_JOINT6')]




#for i in conf.SAFETY_MARGIN:
#    safetyMargin = i;
for s in range(len(conf.SOLVER_TO_INTEGRATE)):   
#    invDynForm = createInvDynFormUtil(q0, v0,invdyn_configs);
#    simulator = createSimulator(q0, v0,simulator_configs);
#    simulator.viewer.setVisibility("floor", "ON" if conf.SHOW_VIEWER_FLOOR else "OFF");
            safetyMargin = conf.SAFETY_MARGIN[s];
            controller_balance[s] = startSimulation(q0, v0, s,safetyMargin);
#    cProfile.run('startSimulation(simulator.q, simulator.dq);');

'''
info = '';
for j in range(len(conf.SOLVER_TO_INTEGRATE)):
    
    #info += "Final com pos solver %d: "%j + str(x_com[j][:,final_time_step[j]].T) + "\n";
    #info += "Final com vel solver %d: "%j + str(dx_com[j][:,final_time_step[j]].T);
    info += " (norm %.3f)\n" % norm(dx_com[j][:,final_time_step[j]]);
    cp_ineq = np.min(np.dot(B_sp, cp[j][:,final_time_step[j]]) + b_sp);
    in_or_out = "outside" if (cp_ineq<0.0) else "inside";
    info += "Final capture point solver %d "%j +str(cp[j][:,final_time_step[j]].T)+" is "+in_or_out+" support polygon %.3f\n"%(cp_ineq);
    info += "Final max joint vel solver %d: %.3f\n"%(j, np.max(np.abs(v[j][:,final_time_step[j]])));
    tfile = open(plot_utils.FIGURE_PATH+conf.TEXT_FILE_NAME, "w")
    tfile.write(info);
info += "***********************************************************************************\n"
print info
'''

if(conf.SAVE_DATA):
    #tfile.write(info);
    #tfile.close(); 
    path = plot_utils.FIGURE_PATH;
    imax = np.max((np.max(i_to_reach),np.max(itmax),np.max(i_to_fall)))  
    
    np.savez_compressed(path+conf.DATA_FILE_NAME, 
                        q=q,
                        v=v,
                        final_time_step=final_time_step,
                        final_time=final_time,
                        falls = falls,
                        time_to_fall=time_to_fall,
                        time_to_reach=time_to_reach,
                        i_to_fall=i_to_fall,
                        i_to_reach=i_to_reach,
                        rh_task_error=rh_task_error,
                        x_com_sim=x_com,
                        cp=cp,
                        cp_ctrl = cp_real,
                        v_cp_robust=v_cp_robust,
                        v_cp_robust_vel=v_cp_robust_vel,
                        v_sp_original=v_sp_original,
                        v_sp_reduced=v_sp_reduced,
                        dt = dt,
                        seed=seed,
                        );

#if(conf.PLAY_MOTION_AT_THE_END):
#    raw_input("Press Enter to play motion in real time...");    
#    for s in conf.SOLVER_TO_INTEGRATE:
#        simulator.viewer.play(q[s][:,:final_time_step[s]], dt, 1.0);
#    print "Computed motion finished";
viewer_utils.ENABLE_VIEWER = True
def play():
    for s in range(len(conf.SOLVER_TO_INTEGRATE)):
        robotwrapper  = RobotWrapper(conf.urdfFileName, conf.model_path, se3.JointModelFreeFlyer())
        viewer1 = viewer_utils.Viewer('viewer',robotwrapper,'robot1');
        #viewer.activateMouseCamera();
        #viewer2 = viewer_utils.Viewer('viewer2',robotwrapper,'robot2');

        viewer1.play(q[s][:,:final_time_step[s]], dt, 1)
        #viewer2.play(q[s][:,:final_time_step[s]], dt, 1)        
