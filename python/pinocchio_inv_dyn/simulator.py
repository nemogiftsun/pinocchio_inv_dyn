import numpy as np
from numpy.linalg import norm
from numpy.random import random
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
from pinocchio.utils import zero as zeros
import pinocchio as se3
from staggered_projections import StaggeredProjections
from constraint_violations import ForceConstraintViolation, PositionConstraintViolation, VelocityConstraintViolation, TorqueConstraintViolation, ConstraintViolationType
from first_order_low_pass_filter import FirstOrderLowPassFilter
import time
from pinocchio.explog import exp
from viewer_utils import Viewer
from sot_utils import compute6dContactInequalities


EPS = 1e-4;

#def zeros(shape):
#    if(isinstance(shape, np.int)):
#        return np.matlib.zeros((shape,1));
#    elif(len(shape)==2):
#        return np.matlib.zeros(shape);
#    raise TypeError("The shape is not an int nor a list of two numbers");

    
class Simulator (object):
    name = ''
    q = None;   # current positions
    v = None;   # current velocities
    
    LCP = None;   # approximate LCP solver using staggered projections
    
    USE_LCP_SOLVER          = False;
    DETECT_CONTACT_POINTS   = False;   ''' True: detect collisions between feet and ground, False: collision is specified by the user '''
    GROUND_HEIGHT           = 0.0;
    LOW_PASS_FILTER_INPUT_TORQUES = False;
    
    ENABLE_FORCE_LIMITS = True;
    ENABLE_TORQUE_LIMITS = True;
    ENABLE_JOINT_LIMITS = True;
    
    ACCOUNT_FOR_ROTOR_INERTIAS = False;

    VIEWER_DT = 0.05;
    DISPLAY_COM = False;
    DISPLAY_CAPTURE_POINT = True;
    COM_SPHERE_RADIUS           = 0.002;
    CAPTURE_POINT_SPHERE_RADIUS = 0.002;
    CONTACT_FORCE_ARROW_RADIUS  = 0.01;
    COM_SPHERE_COLOR            = (1, 0, 0, 1); # red, green, blue, alpha
    CAPTURE_POINT_SPHERE_COLOR  = (0, 1, 0, 1);    
    CONTACT_FORCE_ARROW_COLOR   = (1, 0, 0, 1);
    CONTACT_FORCE_ARROW_SCALE   = 1e-3;
    contact_force_arrow_names = [];  # list of names of contact force arrows
    
    SLIP_VEL_THR = 0.1;
    SLIP_ANGVEL_THR = 0.2;
    NORMAL_FORCE_THR = 5.0;
    JOINT_LIMITS_DQ_THR = 1e-1; #1.0;
    TORQUE_VIOLATION_THR = 1.0;
    DQ_MAX = 9.14286;
                       
    ENABLE_WALL_DRILL_CONTACT = False;
    wall_x = 0.5;
    wall_damping = np.array([30, 30, 30, 0.3, 0.3, 0.3]);
    
    k=0;    # number of contact constraints (i.e. size of contact force vector)
    na=0;   # number of actuated DoFs
    nq=0;   # number of position DoFs
    nv=0;   # number of velocity DoFs
    r=[];   # robot
    
    mu=[];          # friction coefficient (force, moment)
    fMin = 0;       # minimum normal force

    dt = 0;     # time step used to compute joint acceleration bounds
    qMin = [];  # joint lower bounds
    qMax = [];  # joint upper bounds
    
    ''' Mapping between y and tau: y = C*tau+c '''
    C = [];
    c = [];
    #C = zeros((self.nv+self.k+self.na, self.na));
    #c = zeros(self.nv+self.k+self.na);
    
    M = [];         # mass matrix
    Md = [];        # rotor inertia
    h = [];         # dynamic drift
    q = [];
    dq = [];

    x_com = [];     # com 3d position
    dx_com = [];    # com 3d velocity
    ddx_com = [];   # com 3d acceleration
    cp = None;      # capture point
    J_com = [];     # com Jacobian
    Jc = [];        # contact Jacobian
    
    Minv = [];      # inverse of the mass matrix
    Jc_Minv = [];   # Jc*Minv
    Lambda_c = [];  # task-space mass matrix (Jc*Minv*Jc^T)^-1
    Jc_T_pinv = []; # Lambda_c*Jc_Minv
    Nc_T = [];      # I - Jc^T*Jc_T_pinv
    S_T = [];       # selection matrix
    dJc_v = [];     # product of contact Jacobian time derivative and velocity vector: dJc*v
    
    candidateContactConstraints = [];
    rigidContactConstraints = [];   # tasks associated to the contact constraints
    rigidContactConstraints_p = []; # contact points in local frame
    rigidContactConstraints_N = []; # contact normals in local frame
    rigidContactConstraints_fMin = [];  # minimum normal forces
    rigidContactConstraints_mu = [];    # friction coefficients
    rigidContactConstraints_m_in = [];  # number of inequalities
        
    ''' debug variables '''    
    x_c = [];       # contact points
    dx_c = [];      # contact points velocities
    x_c_init = [];  # initial position of constrained bodies    
        
    viewer = None;
    
    def reset(self, t, q, v, dt):
        n = self.nv;
        self.Md = zeros((self.na,self.na)); #np.diag([ g*g*i for (i,g) in zip(INERTIA_ROTOR,GEAR_RATIO) ]); # rotor inertia
        self.q  = np.matrix.copy(q);
        self.v = np.matrix.copy(v);
        self.vOld = np.matrix.copy(v);
        self.dv = zeros(n);
        self.dt = dt;

        self.S_T         = np.zeros((self.na+6,self.na));
        self.S_T[6:, :]  = np.eye(self.na);
        
        self.M          = self.r.mass(self.q);
        self.J_com      = zeros((3,n));

        self.updateInequalityData();

        self.qMin       = self.r.model.lowerPositionLimit;
        self.qMax       = self.r.model.upperPositionLimit;
        self.dqMax      = self.r.model.velocityLimit;
        self.tauMax     = self.r.model.effortLimit;
#        self.ddqMax     = np.array(self.nv*[self.MAX_JOINT_ACC]);
        if(self.freeFlyer):
            self.qMin[:6]   = -1e100;   # set bounds for the floating base
            self.qMax[:6]   = +1e100;
#        self.f_rh        = zeros(6);
#        self.x_c_init       = zeros(k);
#        self.x_c_init[0:3]  = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];
#        self.x_c_init[6:9]  = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3]
#        self.ddx_c_des = zeros(k);
#        self.computeForwardDynamicMapping(t);
        self.INITIALIZE_TORQUE_FILTER = True;
        
    def __init__(self, name, q, v, simulator_configs):
        self.time_step = 0;
        self.DETECT_CONTACT_POINTS = simulator_configs.detectContactPoint;
        self.verb = 0;
        self.name = name;
        self.dt = simulator_configs.dt
        self.mu = simulator_configs.mu;
        self.fMin = simulator_configs.fMin;
        self.freeFlyer = simulator_configs.freeFlyer;
        
        if simulator_configs.r_modified == None:
            if(simulator_configs.freeFlyer):
                self.r = RobotWrapper(simulator_configs.urdfFileName, simulator_configs.mesh_dir, se3.JointModelFreeFlyer());
            else:
                self.r = RobotWrapper(simulator_configs.urdfFileName, simulator_configs.mesh_dir, None);
        else:
            self.r = simulator_configs.r_modified
            
        self.nq = self.r.nq;
        self.nv = self.r.nv;
        self.na = self.nv-6 if self.freeFlyer else self.nv;
        
#        self.candidateContactConstraints = [constr_rf_fr, constr_rf_fl, constr_rf_hr, constr_rf_hl,
#                                            constr_lf_fr, constr_lf_fl, constr_lf_hr, constr_lf_hl];
        self.viewer=Viewer(self.name, self.r);
        
        self.reset(0, q, v, self.dt);
        
        self.LCP = StaggeredProjections(self.nv, self.mu[0], accuracy=EPS, verb=0);
        
        if(self.DISPLAY_COM):
            self.viewer.addSphere('com', self.COM_SPHERE_RADIUS, zeros(3), zeros(3), self.COM_SPHERE_COLOR, 'OFF');
        if(self.DISPLAY_CAPTURE_POINT):
            self.viewer.addSphere('cp', self.CAPTURE_POINT_SPHERE_RADIUS, zeros(3), zeros(3), (0, 1, 0, 1), 'OFF');
            self.viewer.addSphere('cp_real', self.CAPTURE_POINT_SPHERE_RADIUS, zeros(3), zeros(3), (0, 1, 1, 1), 'OFF');        
        
    def updateInequalityData(self):
        c = len(self.rigidContactConstraints);  # number of contacts
        if(self.DETECT_CONTACT_POINTS):
            self.k = c*3;
        else:
            self.k = int(np.sum([con.dim for con in self.rigidContactConstraints]));
        self.Jc         = zeros((self.k,self.nv));
        self.ddx_c_des  = zeros(self.k);
        self.dJc_v      = zeros(self.k);
        self.C           = np.empty((self.nv+self.k+self.na, self.na));
        self.C[self.nv+self.k:,:]     = np.matlib.eye(self.na);
        self.c           = zeros(self.nv+self.k+self.na);
        
        self.Bf = c*[None,]; 
        self.bf = c*[None,];
        for i in range(c):
            (Bfi, bfi) = self.createContactForceInequalities(0.0, self.rigidContactConstraints_mu[i], \
                                                             self.rigidContactConstraints_p[i], self.rigidContactConstraints_N[i], \
                                                             self.rigidContactConstraints[i].framePosition().rotation);
            mask = self.rigidContactConstraints[i]._mask;
            self.Bf[i] = Bfi[:,mask];
            self.bf[i] = bfi;
        
    def setTorqueLowPassFilterCutFrequency(self, fc):
        self.LOW_PASS_FILTER_INPUT_TORQUES = True;
        self.TORQUE_LOW_PASS_FILTER_CUT_FREQUENCY = fc;
        self.INITIALIZE_TORQUE_FILTER = True;
        
        
    ''' Compute the inequality constraints that ensure the contact forces (expressed in world frame)
        are inside the (linearized) friction cones.
        @param fMin Minimum normal force
        @param mu Friction coefficient
        @param contact_points A 3xN matrix containing the contact points expressed in local frame
        @param contact_normals A 3xN matrix containing the contact normals expressed in local frame
        @param oRi Rotation matrix from local to world frame
    '''
    def createContactForceInequalities(self, fMin, mu, contact_points, contact_normals, oRi):
        if(contact_points.shape[1]>1):
            B = -1*compute6dContactInequalities(contact_points.T, contact_normals.T, mu[0]);
            B[:,:3] = np.dot(B[:,:3], oRi.T);
            B[:,3:] = np.dot(B[:,3:], oRi.T);
            b = zeros(B.shape[0]);
        elif(norm(contact_points)<EPS):
            B = zeros((5,6));
            b = zeros(B.shape[0]);
            B[0,0]   = -1;
            B[1,0]   = 1;
            B[2,1]   = -1;
            B[3,1]   = 1;            
            B[:,2]   = mu[0];
            # minimum normal force
            B[-1,2] = 1;
            b[-1]   = -fMin;
        else:
            raise ValueError("Contact with only one point that is not at the origin of the frame: NOT SUPPORTED");

        return (B,b);

    ''' ********** ENABLE OR DISABLE CONTACT CONSTRAINTS ********** '''        

    def addUnilateralContactConstraint(self, constr, contact_points, contact_normals, mu):
        self.rigidContactConstraints        += [constr];
        self.rigidContactConstraints_p      += [contact_points];
        self.rigidContactConstraints_N      += [contact_normals];
        self.rigidContactConstraints_mu     += [mu];
        self.updateInequalityData();
        
    def removeUnilateralContactConstraintByName(self, constr_name):
        if(self.DETECT_CONTACT_POINTS==False):
            found = False;
            for i in range(len(self.rigidContactConstraints)):
                if(self.rigidContactConstraints[i].name==constr_name):
                    del self.rigidContactConstraints[i];
                    found = True;
                    break;
            if(found==False):
                print "SIMULATOR: contact constraint %s cannot be removed!" % constr_name;
            self.updateInequalityData();


    ''' ********** SET ROBOT STATE ********** '''
    
    def setPositions(self, q, updateConstraintReference=True):
#        for i in range(self.nq):
#            if( q[i]>self.qMax[i]+1e-4 ):
#                print "SIMULATOR Joint %d > upper limit, q-qMax=%f deg" % (i,60*(self.q[i]-self.qMax[i]));
#                q[i] = self.qMax[i]-1e-4;
#            elif( q[i]<self.qMin[i]-1e-4 ):
#                print "SIMULATOR Joint %d < lower limit, qMin-q=%f deg" % (i,60*(self.qMin[i]-self.q[i]));
#                q[i] = self.qMin[i]+1e-4;
        self.q = np.matrix.copy(q);
        self.viewer.updateRobotConfig(q);
        
        if(updateConstraintReference):
            self.r.forwardKinematics(q);
            for c in self.rigidContactConstraints:
                Mref = self.r.position(q, c._link_id, update_geometry=False);
                c.refTrajectory.setReference(Mref);
#            if(self.DETECT_CONTACT_POINTS==False):
#                t = self.rigidContactConstraints[0].opPointModif.position.time;
#                for c in self.rigidContactConstraints:
#                    c.opPointModif.position.recompute(t+1);
#                    c.ref = c.opPointModif.position.value;
#            else:
#                self.rigidContactConstraints = []
#                t = self.candidateContactConstraints[0].opPointModif.position.time;
#                for c in self.candidateContactConstraints:
#                    c.opPointModif.position.recompute(t+1);
#                    if(c.opPointModif.position.value[2][3] < self.GROUND_HEIGHT):
#                        c.ref = c.opPointModif.position.value;
#                        self.rigidContactConstraints = self.rigidContactConstraints + [c,];
#                        print "[SIMULATOR::setPositions] Collision detected for constraint %s" % c.name;
#                        
#            self.x_c_init[0:3]  = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];
#            self.x_c_init[6:9]  = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];
        return self.q;
    
    def setVelocities(self, v):
        self.v = np.matrix.copy(v);
        return self.v;


    def computeForwardDynamicMapping(self, t):
        n = self.na;
        self.t = t;

        ''' Set the position and velocity input signals of the dynamic entity.
            Actually this should be useless because those signals are already plugged
            to the output signals of the device entity. However, due to a small error
            introduced in the conversion python-C++ when setting the signal values,
            it is better to set these signals so that the same errors occur in the simulator
            and in the controller.
        '''
        q = self.q;
        v = self.v;
        k = self.k;
        nv = self.nv;
        self.r.computeAllTerms(q, v);
        self.r.framesKinematics(q);                
        i = 0;
        if(self.DETECT_CONTACT_POINTS==True):
            raise ValueError("[simulator] Contact point detection not implemented yet");
            ''' COLLISION DETECTION '''
            constraintsChanged = False;
            oldConstraints = list(self.rigidContactConstraints); # copy list
            for c in self.candidateContactConstraints:
                # do not update task values here because task ref may change in this for loop
                c.task.jacobian.recompute(t);
                c.task.Jdot.recompute(t);
                c.opPointModif.position.recompute(t);
                if(c in self.rigidContactConstraints):
                    if(c.opPointModif.position.value[2][3] > self.GROUND_HEIGHT):
                        j = oldConstraints.index(c);
                        if(self.f[3*j+2]<EPS):
                            self.rigidContactConstraints.remove(c);
                            constraintsChanged = True;
                        elif(self.verb>0):
                            print "Collision lost for constraint %s, but I'm gonna keep it because previous normal force was %f" % (c.name, self.f[3*j+2]);                        
                else:
                    if(c.opPointModif.position.value[2][3] <= self.GROUND_HEIGHT):
                        c.ref = c.opPointModif.position.value;
                        self.rigidContactConstraints.append(c);                        
                        constraintsChanged = True;
                        if(self.verb>0):
                            print "Contact detected for constraint %s, pos %.3f %.3f %.3f" % (c.name
                                                                                              ,c.opPointModif.position.value[0][3]
                                                                                              ,c.opPointModif.position.value[1][3]
                                                                                              ,c.opPointModif.position.value[2][3]);
            if(constraintsChanged):
                self.updateInequalityData();
            for constr in self.rigidContactConstraints:
                # now update task values
                constr.task.task.recompute(t);
                self.Jc[i*3:i*3+3,:]      = np.array(constr.task.jacobian.value)[0:3,:];
                self.dJc_v[i*3:i*3+3]     = np.dot(constr.task.Jdot.value, self.dq)[0:3];
                # do not compensate for drift in tangential directions
                self.ddx_c_des[i*3:i*3+2] = np.zeros(2);
                self.ddx_c_des[i*3+2]     = np.array(constr.task.task.value)[2];
#                self.ddx_c_des[i*3:i*3+2] = - np.dot(np.array(constr.task.Jdot.value)[0:2,:], self.dq);
#                self.ddx_c_des[i*3+2:i*3+3] = np.array(constr.task.task.value)[2] - \
#                                              np.dot(np.array(constr.task.Jdot.value)[2,:], self.dq);
                i = i+1;
        else:
            for constr in self.rigidContactConstraints:
                dim = constr.dim
                (self.Jc[i:i+dim,:], self.dJc_v[i:i+dim], self.ddx_c_des[i:i+dim]) = constr.dyn_value(t, q, v, local_frame=False);
                i += dim;            

        self.M        = self.r.mass(self.q, update_kinematics=False);
        self.Ag       = self.r.momentumJacobian(self.q, self.v);
        if(self.ACCOUNT_FOR_ROTOR_INERTIAS):
            if(self.freeFlyer):
                self.M[6:,6:]   += self.Md;
            else:
                self.M   += self.Md;                        
        self.h        = self.r.bias(self.q,self.v, update_kinematics=False);             
        self.dJc_v      -= self.ddx_c_des;
        self.Minv        = np.linalg.inv(self.M);
        if(self.k>0):
            self.Jc_Minv     = np.dot(self.Jc, self.Minv);
            self.Lambda_c    = np.linalg.inv(np.dot(self.Jc_Minv, self.Jc.T) + 1e-10*np.matlib.eye(self.k));
            self.Jc_T_pinv   = np.dot(self.Lambda_c, self.Jc_Minv);
            self.Nc_T        = np.matlib.eye(self.nv) - np.dot(self.Jc.T, self.Jc_T_pinv);
            self.dx_c        = np.dot(self.Jc, self.v);
            # Compute C and c such that y = C*tau + c, where y = [dv, f, tau]
            self.C[0:nv,:]      = np.dot(self.Minv, np.dot(self.Nc_T, self.S_T));
            self.C[nv:nv+k,:]   = -np.dot(self.Jc_T_pinv, self.S_T);
            self.C[nv+k:,:]     = np.matlib.eye(self.na);
            self.c[0:nv]        = - np.dot(self.Minv, (np.dot(self.Nc_T,self.h) + np.dot(self.Jc.T, np.dot(self.Lambda_c, self.dJc_v - self.ddx_c_des))));
            self.c[nv:nv+k]     = np.dot(self.Lambda_c, (np.dot(self.Jc_Minv, self.h) - self.dJc_v + self.ddx_c_des));
        else:
            self.Nc_T        = np.matlib.eye(self.nv);
            self.C[0:nv,:]      = np.dot(self.Minv, self.S_T);
            self.C[nv:,:]       = np.matlib.eye(self.na);
            self.c[0:nv]        = - np.dot(self.Minv, self.h);
            
        self.h_hat = np.copy(self.h);
        
        '''  Compute capture point '''
        self.x_com    = self.r.com(self.q, update_kinematics=False);
        self.J_com    = self.r.Jcom(self.q, update_kinematics=False);        
        self.dx_com     = np.dot(self.J_com, self.v);                
        com_z           = self.x_com[2]; #-np.mean(self.contact_points[:,2]);
        if(com_z>0.0):
            self.cp         = self.x_com[:2] + self.dx_com[:2]/np.sqrt(9.81/com_z);
        else:
            self.cp = zeros(2);           

        return (self.C, self.c);


    def integrate(self,t, dt, tau):
#        ConstraintViolation = enum(none=0, force=1, position=2, torque=3);
        res = []
        nv = self.nv;
        k = self.k;
        self.dt = dt
        self.computeForwardDynamicMapping(t);
        if(self.LOW_PASS_FILTER_INPUT_TORQUES):
            if(self.INITIALIZE_TORQUE_FILTER):
                self.torqueFilter = FirstOrderLowPassFilter(self.dt, self.TORQUE_LOW_PASS_FILTER_CUT_FREQUENCY, tau);
                self.INITIALIZE_TORQUE_FILTER = False;
            self.tau = self.torqueFilter.filter_data(np.copy(tau));
        else:
            self.tau = tau;
        
        ''' Check for violation of torque limits '''
        for i in range(self.na):
            if( self.tau[i]>self.tauMax[6+i]+self.TORQUE_VIOLATION_THR):                    
                res = res + [TorqueConstraintViolation(self.t*self.dt, i, self.tau[i])];
                if(self.ENABLE_TORQUE_LIMITS):
                    self.tau[i] = self.tauMax[6+i];
            elif(self.tau[i]<-self.tauMax[6+i]-self.TORQUE_VIOLATION_THR):
                res = res + [TorqueConstraintViolation(self.t*self.dt, i, self.tau[i])];
                if(self.ENABLE_TORQUE_LIMITS):
                    self.tau[i] = -self.tauMax[6+i];
        #self.qOld  = np.copy(self.q);
        #self.dqOld = np.copy(self.dq);
        
        if(self.USE_LCP_SOLVER):
            raise ValueError("[simulator] Simulation with LCP solver not implemented yet");
            '''
            self.Jc_list = len(self.rigidContactConstraints)*[None,];
            self.dJv_list = len(self.rigidContactConstraints)*[None,];
            i = 0;
            for constr in self.rigidContactConstraints:
                self.Jc_list[i]      = np.array(constr.task.jacobian.value)[0:3,:];
                self.dJv_list[i]     = np.dot(constr.task.Jdot.value, self.dq)[0:3] - np.array(constr.task.task.value)[0:3];
                i = i+1;
                
            (v, self.f) = self.LCP.simulate(self.dq, self.M, self.h_hat, self.tau, dt, self.Jc_list, self.dJv_list, maxIter=None, maxTime=100.0);
#            for i in range(len(self.rigidContactConstraints)):
#                if(self.f[3*i+2]<1e-2):
#                    print "[SIMULATOR] Contact %s has normal force %f" % (self.rigidContactConstraints[i].name, self.f[3*i+2]);
            self.dv = (v-self.dqOld)/dt;  
            '''
        else:
            ''' compute accelerations and contact forces from joint torques '''
            y = np.dot(self.C, self.tau) + self.c;
            self.dv = y[:nv];
            self.f  = y[nv:nv+k];
            
        # Check that contact forces are inside friction cones
#        ii = 0;
#        for i in range(len(self.rigidContactConstraints)):
#            dim = self.rigidContactConstraints[i].dim;
#            f_constr = self.Bf[i]*self.f[ii:ii+dim] + self.bf[i];
#            ii += dim;
#            if(f_constr<-EPS).any():
#                if(self.verb>0):
#                    print "[SIMULATOR] Friction cone violation %s: %f" % (self.rigidContactConstraints[i].name, np.min(f_constr));
#                res = res + [ForceConstraintViolation(self.t*self.dt, self.rigidContactConstraints[i].name, self.f[ii:ii+dim], zeros(6))];                        

        return res+self.integrateAcc(t,dt,self.dv,True)
    
    def integrateAcc(self, t, dt, dv,updateViewer=True):
        res = []
        self.t = t;
        self.time_step += 1;
        self.dv = np.matrix.copy(dv);
        
        if(abs(norm(self.q[3:7])-1.0) > EPS):
            print "SIMULATOR ERROR Time %.3f "%t, "norm of quaternion is not 1=%f" % norm(self.q[3:7]);
            
        ''' Integrate velocity and acceleration '''
        self.q  = se3.integrate(self.r.model, self.q, dt*self.v);
        self.v += dt*self.dv;
        
        ''' Compute CoM acceleration '''
#        self.ddx_com = np.dot(self.J_com, self.dv) + np.dot(self.dJ_com, self.v);
        
        ''' DEBUG '''
#        q_pino = sot_2_pinocchio(self.q);
#        v_pino = sot_2_pinocchio_vel(self.dq);
#        g_pino = np.array(self.viewer.robot.gravity(q_pino));
#        h_pino = np.array(self.viewer.robot.biais(q_pino, v_pino));
#        self.viewer.robot.forwardKinematics(q_pino);
#        dJcom_dq = self.viewer.robot.data.oMi[1].rotation*(h_pino[:3] - g_pino[:3]) / self.M[0,0];
#        ddx_com_pino = np.dot(self.J_com, self.dv) + dJcom_dq.reshape(3);
#        
#        self.r.dynamic.position.value = tuple(self.q);
#        self.r.dynamic.velocity.value = tuple(self.dq);
#        self.r.dynamic.com.recompute(self.t+1);
#        self.r.dynamic.Jcom.recompute(self.t+1);
#        new_x_com = np.array(self.r.dynamic.com.value);
#        new_J_com = np.array(self.r.dynamic.Jcom.value);
#        new_dx_com = np.dot(new_J_com, self.dq);
#        new_dx_com_int = self.dx_com + dt*self.ddx_com;
        #if(np.linalg.norm(new_dx_com - new_dx_com_int) > 1e-3):
        #    print "Time %.3f ERROR in integration of com acc"%t, "%.3f"%np.linalg.norm(new_dx_com - new_dx_com_int), new_dx_com, new_dx_com_int;
            
            
        ''' END DEBUG '''
        
#        if(self.DETECT_CONTACT_POINTS==True):
#            ''' compute feet wrenches '''
#            self.w_rf = zeros(6);
#            self.w_lf = zeros(6);
#            X = zeros((6,3));
#            X[:3,:] = np.identity(3);
#            i=0;
#            for c in self.rigidContactConstraints:
#                X[3:,:] = crossMatrix(np.array(c.opmodif)[:3,3] - np.array(H_FOOT_2_SOLE)[:3,3]);
#                if('c_rf' in c.name):
#                    self.w_rf += np.dot(X, self.f[i*3:i*3+3]);
#                elif('c_lf' in c.name):
#                    self.w_lf += np.dot(X, self.f[i*3:i*3+3]);
#                i += 1;
#        else:
#            self.w_rf = f[:6];
#            self.w_lf = f[6:];
#                
#        ''' check for slippage '''
#        if(self.DETECT_CONTACT_POINTS==True):
#            if((self.support_phase==Support.left or self.support_phase==Support.double) and self.areAllFootCornersInContact('c_lf')):
#                v = self.getLeftFootVel(self.t+1);
#                w = self.w_lf;
#                if((np.linalg.norm(v[:3])>self.SLIP_VEL_THR or np.linalg.norm(v[3:])>self.SLIP_ANGVEL_THR) and w[2]>self.NORMAL_FORCE_THR):
#                    res += [ForceConstraintViolation(self.t*self.dt, 'c_lf', w, v)];
#            if((self.support_phase==Support.right or self.support_phase==Support.double) and self.areAllFootCornersInContact('c_rf')):
#                v = self.getRightFootVel(self.t+1);
#                w = self.w_rf;
#                if((np.linalg.norm(v[:3])>self.SLIP_VEL_THR or np.linalg.norm(v[3:])>self.SLIP_ANGVEL_THR) and w[2]>self.NORMAL_FORCE_THR):
#                    res += [ForceConstraintViolation(self.t*self.dt, 'c_rf', w, v)];
#        
#        ''' Check for violation of force limits'''
#        mu = self.mu;
#        if(self.DETECT_CONTACT_POINTS==True):
#            for contactName in ['right_foot','left_foot']:
#                if(contactName=='right_foot'):
#                    fx=self.w_rf[0]; fy=self.w_rf[1]; fz=self.w_rf[2];
#                else:
#                    fx=self.w_lf[0]; fy=self.w_lf[1]; fz=self.w_lf[2];
#                if(fx+mu[0]*fz<-2*EPS or -fx+mu[0]*fz<-2*EPS):
#                    if(fz!=0.0 and self.verb>0):
#                        print "SIMULATOR: friction cone %s x violated, fx=%f, fz=%f, fx/fz=%f" % (contactName,fx,fz,fx/fz);
#                if(fy+mu[0]*fz<-2*EPS or -fy+mu[0]*fz<-2*EPS):
#                    if(fz!=0.0 and self.verb>0):
#                        print "SIMULATOR: friction cone %s y violated, fy=%f, fz=%f, fy/fz=%f" % (contactName,fy,fz,fy/fz);
#                if(fz<-2*EPS and self.verb>0):
#                    print "SIMULATOR: normal force %s z negative, fz=%f" % (contactName,fz);
#        else:                

        ''' check for violations of joint limits '''
        ind_vel = np.where(np.abs(self.v) > self.DQ_MAX)[0].squeeze();
        self.ind_vel = np.array([ind_vel]) if len(ind_vel.shape)==0 else ind_vel;
        
        if ind_vel.tolist()[0] != []:
            for i in ind_vel.tolist()[0]:
                res = res + [VelocityConstraintViolation(self.t*self.dt, i-7, self.v[i], self.dv[i])];
                if(self.verb>0):
                    print "[SIMULATOR] %s" % (res[-1].toString());
                if(self.ENABLE_JOINT_LIMITS):
                    self.v[i] = self.DQ_MAX if (self.v[i]>0.0) else -self.DQ_MAX;
        
        
        ind_pos_ub = (self.q[7:]>self.qMax[7:]+EPS).A.squeeze();
        ind_pos_lb = (self.q[7:]<self.qMin[7:]-EPS).A.squeeze();
        for i in np.where(ind_pos_ub)[0]:
            res = res + [PositionConstraintViolation(self.t*self.dt, i, self.q[7+i], self.v[6+i], self.dv[6+i])];
            if(self.verb>0):
                print "[SIMULATOR] %s" % (res[-1].toString());
        for i in np.where(ind_pos_lb)[0]:
            res = res + [PositionConstraintViolation(self.t*self.dt, i, self.q[7+i], self.v[6+i], self.dv[6+i])];
            if(self.verb>0):
                print "[SIMULATOR] %s" % (res[-1].toString());
                
        if(self.ENABLE_JOINT_LIMITS):
            self.q[7:][ind_pos_ub] = self.qMax[7:][ind_pos_ub];
            self.v[6:][ind_pos_ub] = 0.0;
            self.q[7:][ind_pos_lb] = self.qMin[7:][ind_pos_lb];
            self.v[6:][ind_pos_lb] = 0.0;
        
        if(updateViewer and self.time_step%int(self.VIEWER_DT/dt)==0):
            self.viewer.updateRobotConfig(self.q);
#            if(self.DISPLAY_COM):
#                self.viewer.updateObjectConfig('com', (self.x_com[0], self.x_com[1], 0, 0,0,0,1));
#            if(self.DISPLAY_CAPTURE_POINT):
#                self.viewer.updateObjectConfig('cp', (self.cp[0], self.cp[1], 0, 0,0,0,1));
            
        return res;
        
    def updateComPositionInViewer(self, com):
        assert np.asarray(com).squeeze().shape[0]==3, "com should be a 3x1 matrix"
        com = np.asarray(com).squeeze();
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            if(self.DISPLAY_COM):
                self.viewer.updateObjectConfig('com', (com[0], com[1], com[2], 0.,0.,0.,1.));
    
    def updateCapturePointPositionInViewer(self, cp,cp_real):
        assert np.asarray(cp).squeeze().shape[0]==2, "capture point should be a 2d vector"
        cp = np.asarray(cp).squeeze();
        cp_real = np.asarray(cp_real).squeeze();
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            if(self.DISPLAY_CAPTURE_POINT):
                self.viewer.updateObjectConfig('cp', (cp[0], cp[1], -0.02, 0.,0.,0.,1.));
                self.viewer.updateObjectConfig('cp_real', (cp_real[0], cp_real[1], -0.02, 0.,0.,0.,1.));            
    

    ''' Update the arrows representing the specified contact forces in the viewer.
        If the arrows have not been created yet, it creates them.
        If a force arrow that is currently displayed does not appear in the specified
        list, the arrow visibility is set to OFF.
        @param contact_names A list of contact names
        @param contact_points A list of contact points (i.e. 3x1 numpy matrices)
        @param contact_forces A list of contact forces (i.e. 3x1 numpy matrices)
    '''
    def updateContactForcesInViewer(self, contact_names, contact_points, contact_forces):
        if(self.time_step%int(self.VIEWER_DT/self.dt)==0):
            for (name, p, f) in zip(contact_names, contact_points, contact_forces):
                if(name not in self.contact_force_arrow_names):
                    self.viewer.addArrow(name, self.CONTACT_FORCE_ARROW_RADIUS, p, p+self.CONTACT_FORCE_ARROW_SCALE*f[:3], self.CONTACT_FORCE_ARROW_COLOR);
                    self.viewer.setVisibility(name, "ON");
                    self.contact_force_arrow_names += [name];
                else:
                    self.viewer.moveArrow(name, p, p+self.CONTACT_FORCE_ARROW_SCALE*f[:3]);
                    
            for name in self.viewer.arrow_radius:
                if(name not in contact_names):
                    self.viewer.setVisibility(name, "OFF");
                    
            self.contact_force_arrow_names = list(contact_names);

    def updateContactForcesInViewerFromConstraints(self, contact_constraints, f):
        contact_names  = [con.name for con in contact_constraints];
        contact_sizes  = [con.dim for con in contact_constraints];
        contact_size_cum = [int(np.sum(contact_sizes[:ii])) for ii in range(len(contact_sizes))];
        contact_points = [con.framePosition().translation for con in contact_constraints];
        contact_forces = [ f[ii:ii+3] for ii in contact_size_cum];
        return self.updateContactForcesInViewer(contact_names, contact_points, contact_forces);

            