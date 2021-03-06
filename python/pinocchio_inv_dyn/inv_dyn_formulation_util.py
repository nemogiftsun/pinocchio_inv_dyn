import numpy as np
from numpy.linalg import norm
from numpy.random import random
from pinocchio_inv_dyn.geom_utils import check_point_polytope
from pinocchio_inv_dyn.polytope_conversion_utils import poly_face_to_span,poly_span_to_face,cone_span_to_face
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import pinocchio as se3
from pinocchio.utils import zero as zeros
from pinocchio.utils import skew
from acc_bounds_util_multi_dof import computeAccLimits
from sot_utils import compute6dContactInequalities, crossMatrix
from first_order_low_pass_filter import FirstOrderLowPassFilter
from convex_hull_util import compute_convex_hull, plot_convex_hull
from geom_utils import plot_polytope
from multi_contact.utils import compute_GIWC, compute_support_polygon
EPS = 1e-4;
    
    
class InvDynFormulation (object):
    name = '';
    verb = 0;
    
    ENABLE_JOINT_LIMITS         = True;
    ENABLE_CAPTURE_POINT_LIMITS = False;
    INCLUDE_VEL_UNCERTAINTIES = False;
    ENABLE_CAPTURE_POINT_LIMITS_ROBUST = False;
    ENABLE_TORQUE_LIMITS        = True;
    ENABLE_FORCE_LIMITS         = True;
    POLYTOPE_MARGIN             = 0;

    #MAX_COM_ERROR = 0.0
    #MAX_MASS_ERROR = 0.0
    #MAX_INERTIA_ERROR = 0.0
    
    USE_JOINT_VELOCITY_ESTIMATOR = False;
    BASE_VEL_FILTER_CUT_FREQ = 5;
    JOINT_VEL_ESTIMATOR_DELAY = 0.02;
    COMPUTE_SUPPORT_POLYGON = True;
    
    ACCOUNT_FOR_ROTOR_INERTIAS = True;
    
    JOINT_FRICTION_COMPENSATION_PERCENTAGE = 1.0;
    MAX_JOINT_ACC = 100.0;      # maximum acceleration upper bound
    MAX_MIN_JOINT_ACC = 10.0;   # maximum acceleration lower bound
    JOINT_POS_PREVIEW = 1; # preview window to convert joint pos limits into joint acc limits
    JOINT_VEL_PREVIEW = 1;  # preview window to convert joint vel limits into joint acc limits
    
    na=0;    # number of actuated joints
    nq=0;    # number of position DoFs
    nv=0;   # number of velocity DoFs
    m_in=0; # number of inequalities
    k=0;    # number of contact constraints (i.e. size of contact force vector)
    
    ind_force_in = [];  # indeces of force inequalities
    ind_acc_in = [];    # indeces of acceleration inequalities
    ind_cp_in = [];     # indeces of capture point inequalities
    
    tauMax=[];  # torque limits

    dt = 0;     # time step used to compute joint acceleration bounds
    qMin = [];  # joint lower bounds
    qMax = [];  # joint upper bounds
    dqMax = []; # max joint velocities
    ddqMax = []; # max joint accelerations
    
    ddqMaxFinal = [];   # max joint accelerations resulting from pos/vel/acc limits
    ddqMinFinal = [];   # min joint accelerations resulting from pos/vel/acc limits

    ''' Classic inverse dynamics formulation
            minimize    ||A*y-a||^2
            subject to  B*y+b >= 0
                        dynamics(y) = 0
        where y=[dv, f, tau]
    '''
    A = [];
    a = [];
    B = [];
    b = [];
    
    ''' Mapping between y and tau: y = C*tau+c '''
    C = [];
    c = [];

    ''' Reformulation of the inverse dynamics optimization problem
        in terms of tau only:
            minimize    ||D*tau-d||^2
            subject to  G*tau+g >= 0
    '''
    D = [];
    d = [];
    G = [];
    g = [];
    
    M = [];         # mass matrix
    h = [];         # dynamic drift
    q = [];
    v = [];
    
    x_com = [];     # com 3d position
    dx_com = [];    # com 3d velocity
    ddx_com = [];   # com 3d acceleration
    cp = [];        # capture point

    J_com = [];     # com Jacobian
    Jc = [];        # contact Jacobian
    x_c = [];       # contact points
    dx_c = [];      # contact points velocities
    
    Minv = [];      # inverse of the mass matrix
    Jc_Minv = [];   # Jc*Minv
    Lambda_c = [];  # task-space mass matrix (Jc*Minv*Jc^T)^-1
    Jc_T_pinv = []; # Lambda_c*Jc_Minv
    Nc_T = [];      # I - Jc^T*Jc_T_pinv
    S_T = [];       # selection matrix
    dJc_v = [];     # product of contact Jacobian time derivative and velocity vector: dJc*v
    
    rigidContactConstraints = [];   # tasks associated to the contact constraints
    rigidContactConstraints_p = []; # contact points in local frame
    rigidContactConstraints_N = []; # contact normals in local frame
    rigidContactConstraints_fMin = [];  # minimum normal forces
    rigidContactConstraints_mu = [];    # friction coefficients
    rigidContactConstraints_m_in = [];  # number of inequalities
    bilateralContactConstraints = [];

    V = None
    N = None
    vcom = []
    
    tasks = [];
    task_weights = [];
    
    B_sp = None;     # 2d support polygon: B_sp*x + b_sp >= 0
    b_sp = None;
    support_polygon_computed = False;
    
    contact_points = None;  # 3xN matrix containing the contact points in world frame
    contact_normals = None; # 3xN matrix containing the contact normals in world frame

    
    def updateInequalityData(self, updateConstrainedDynamics=True):
        self.updateSupportPolygon();
        self.m_in = 0;                              # number of inequalities
        c = len(self.rigidContactConstraints);      # number of unilateral contacts
        self.k = int(np.sum([con.dim for con in self.rigidContactConstraints]));
        self.k += int(np.sum([con.dim for con in self.bilateralContactConstraints]));
        if(self.ENABLE_FORCE_LIMITS):
            self.rigidContactConstraints_m_in = np.matlib.zeros(c, np.int);
            Bf = zeros((0,self.k));
            bf = zeros(0);
            ii = 0;
            for i in range(c):
                (Bfi, bfi) = self.createContactForceInequalities(self.rigidContactConstraints_fMin[i], self.rigidContactConstraints_mu[i], \
                                                                 self.rigidContactConstraints_p[i], self.rigidContactConstraints_N[i], \
                                                                 self.rigidContactConstraints[i].framePosition().rotation);
                self.rigidContactConstraints_m_in[0,i] = Bfi.shape[0];
                tmp = zeros((Bfi.shape[0], self.k));
                dim = self.rigidContactConstraints[i].dim;
                mask = self.rigidContactConstraints[i]._mask;
                tmp[:,ii:ii+dim] = Bfi[:,mask];
                ii += dim;
                Bf = np.vstack((Bf, tmp));
                bf = np.vstack((bf, bfi));
            self.ind_force_in = range(self.m_in, self.m_in + np.sum(self.rigidContactConstraints_m_in));
            self.m_in += np.sum(self.rigidContactConstraints_m_in);
        else:
            self.ind_force_in = [];

        if(self.ENABLE_JOINT_LIMITS):
            self.ind_acc_in = range(self.m_in, self.m_in+2*self.na);
            self.m_in += 2*self.na;
        else:
            self.ind_acc_in = [];
            
        if(self.ENABLE_TORQUE_LIMITS):
            self.lb = -self.tauMax;
            self.ub = self.tauMax;
        else:
            self.lb = zeros(self.na) - 1e100;
            self.ub = zeros(self.na) + 1e100;
            
        if(self.ENABLE_CAPTURE_POINT_LIMITS or self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST):
            self.ind_cp_in = range(self.m_in, self.m_in+self.b_sp.size);
            self.m_in += self.b_sp.size;
        else:
            self.ind_cp_in = [];
            
        # resize all data that depends on k
        self.B          = zeros((self.m_in, self.nv+self.k+self.na));
        self.b          = zeros(self.m_in);
        self.Jc         = zeros((self.k,self.nv));
        self.dJc_v      = zeros(self.k);
        self.dx_c       = zeros(self.k);
        self.ddx_c_des  = zeros(self.k);
        self.Jc_Minv    = zeros((self.k,self.nv));
        self.Lambda_c   = zeros((self.k,self.k));
        self.Jc_T_pinv  = zeros((self.k,self.nv));
        self.C           = zeros((self.nv+self.k+self.na, self.na));
        self.c           = zeros(self.nv+self.k+self.na);
        
        if(self.ENABLE_FORCE_LIMITS and self.k>0):
            self.B[self.ind_force_in, self.nv:self.nv+self.k] = Bf;
            self.b[self.ind_force_in] = bf;
            #print "Contact inequality constraints:\n", self.B[self.ind_force_in, self.nv:self.nv+self.k], "\n", bf.T;
        
        if(updateConstrainedDynamics):
            self.updateConstrainedDynamics();
        
    
    def __init__(self, name, q, v,invdyn_configs):
        self.firstTime = True; 
        if(invdyn_configs.freeFlyer):
            self.r = RobotWrapper(invdyn_configs.urdfFileName, invdyn_configs.mesh_dir, root_joint=se3.JointModelFreeFlyer());
        else:
            self.r = RobotWrapper(invdyn_configs.urdfFileName, invdyn_configs.mesh_dir, None);
        self.freeFlyer = invdyn_configs.freeFlyer;
        self.nq = self.r.nq;
        self.nv = self.r.nv;
        self.na = self.nv-6 if self.freeFlyer else self.nv;
        self.k = 0;        # number of constraints
        self.dt = invdyn_configs.dt;
        self.t = 0.0;
        self.name = name;
        self.ACCOUNT_FOR_ROTOR_INERTIAS = invdyn_configs.ACCOUNT_FOR_ROTOR_INERTIAS
        if self.ACCOUNT_FOR_ROTOR_INERTIAS ==  True:
            self.Md = np.diag([ g*g*i for (i,g) in zip(invdyn_configs.INERTIA_ROTOR[6:],invdyn_configs.GEAR_RATIO[6:]) ]); # rotor inertia
        else:
            self.Md = zeros((self.na,self.na))
            
        self.dJ_com      = zeros((3,self.nv));
        ''' create low-pass filter for base velocities '''
        self.baseVelocityFilter = FirstOrderLowPassFilter(self.dt, self.BASE_VEL_FILTER_CUT_FREQ , zeros(6));            
        if(invdyn_configs.freeFlyer):
            self.S_T         = zeros((self.nv,self.na));
            self.S_T[6:, :]  = np.matlib.eye(self.na);
        else:
            self.S_T    = np.matlib.eye(self.na);
        self.Nc_T       = np.matlib.eye(self.nv);
    
        self.qMin       = self.r.model.lowerPositionLimit;
        self.qMax       = self.r.model.upperPositionLimit;
        self.dqMax      = self.r.model.velocityLimit;
        # lets check : need to be removed
        #self.dqMax      = np.matlib.ones((36,1))*9.14286
        self.ddqMax     = zeros(self.na); 
        self.ddqStop    = zeros(self.na);
        if(self.freeFlyer):
            self.qMin[:6]   = -1e100;   # set bounds for the floating base
            self.qMax[:6]   = +1e100;
            self.tauMax     = self.r.model.effortLimit[6:];
        else:
            self.tauMax     = self.r.model.effortLimit;
                        
        self.contact_points = zeros((0,3));
        self.updateInequalityData(updateConstrainedDynamics=False);
        self.setNewSensorData(0, q, v);   
        self.inertiaError = invdyn_configs.inertiaError;
        self.V,self.N = invdyn_configs.vcom,invdyn_configs.ncom
    #WORK NEED TO BE DONE#    
#    def computeGlobalCOMPolytope(self,V,N):
#        # initialize params
#        joint_ns = 31     
#        #flagged_ns = len(self.r.mass)
#        vlinks_sum= np.matlib.zeros((2,self.B_sp.shape[0]))
#        vlinks_vel_sum= np.matlib.zeros((2,self.B_sp.shape[0]))        
#        # works only when map_index shape equals the number of polytope
#        totalmass = 0
#        for i in range(joint_ns):
#            idx = self.r.model.names[i+1]
#            fid = self.getFrameId(idx)  
#            a = int(np.sum(N[0,0:i]))
#            b = int(N[0,i]+a) 
#            vlink = np.matlib.zeros((3,int(N[0,i])))
#            m = 0
#            for j in range(a,b):
#                vlink[:,m] = self.r.data.oMi[i+1].act(V[:,j])                
#                m += 1
#            while True:
#                try:
#                    Av,bv = poly_span_to_face(np.asarray(vlink))
#                except:
#                    r = np.ones((vlink.shape[0],vlink.shape[1]))*1e-1;
#                    vlink = vlink +r    
#                    print '---Exception----'
#                    print i
#                    print '----------------'
#                    continue     
#                break 
#            # compute velocity
#            #V_LINK_VEL = np.matlib.zeros((3,V_link.shape[1]))    
#            vlink_vel = np.matlib.zeros((3,vlink.shape[1]))   
#            for pt in range(vlink.shape[1]): 
#                p = skew(vlink[:,pt])
#                vlink_jc = self.r.frameJacobian(self.q,fid)[0:3,:]- p*self.r.frameJacobian(self.q,fid)[3:6,:]
#                vlink_vel[:,pt] = np.dot(vlink_jc,self.v)
#                 
#            # multiply with mass
#            mass_links_bound = [self.r.model.inertias[i+1].mass-(self.r.model.inertias[i+1].mass*self.MAX_MASS_ERROR),self.r.model.inertias[i+1].mass+(self.r.model.inertias[i+1].mass*self.MAX_MASS_ERROR)]  
#            V_mlb = mass_links_bound[0]*vlink
#            V_mhb = mass_links_bound[1]*vlink
#            V_vel_mlb = mass_links_bound[0]*vlink_vel
#            V_vel_mhb = mass_links_bound[1]*vlink_vel            
#            V_link = np.hstack((V_mhb,V_mlb))
#            V_link_vel =  np.hstack((V_vel_mhb,V_vel_mlb))
#            ##NEW##                
#           #V_link_vel = np.matlib.zeros(V_link.shape)         
#            V_link_t = se3.SE3.Identity()
#              
#            ##NEW##
#            #Alink,blink = compute_convex_hull(V_link[0:3,:])
#            #V_link = poly_face_to_span(-Alink,blink)
#            #V_link = np.vstack((V_link,np.zeros((0,V_link.shape[1]))))
#            ##NEW##        
#            '''
#            V_link_frame = np.copy(V_link)
#            for col in range(V_link.shape[1]): 
#                #print np.asmatrix(V_link[:,col])
#                #print V_link_t.translation[0:2]
#                V_link_t.translation = np.asmatrix(V_link[:,col]).T
#                V_link_frame[:,col] = (self.r.data.oMi[i+1].inverse()*V_link_t).translation.T  
#            '''  
# 
##            for pt in range(V_link.shape[1]): 
##                p = skew(np.hstack((V_link[:,pt],0)))
##                V_LINK_JC = self.r.frameJacobian(self.q,fid)[0:3,:]+ p*self.r.frameJacobian(self.q,fid)[3:6,:]
##                V_LINK_VEL[:,pt] = np.dot(V_LINK_JC,self.v)                             
##            frameVel = self.r.frameVelocity(fid)
##            V_link =  np.vstack((V_link,np.zeros((1,V_link.shape[1]))))
##            #V_link_vel = frameVel.linear + skew(frameVel.angular)*V_link
##            V_link_vel = V_LINK_VEL
#            print V_link_vel
#            print V_link
#            while True:
#                try:
#                    Alink,blink = compute_convex_hull(V_link[:,:])
#                    V_link = poly_face_to_span(-Alink,blink)
#                    print V_link
#                    Alink,blink = compute_convex_hull(V_link_vel[:,:])
#                    V_link_vel = poly_face_to_span(-Alink,blink)                    
#                    #V_link = np.vstack((V_link,np.zeros((0,V_link.shape[1]))))
#                    ##NEW##        
#                    '''
#                    V_link_frame = np.copy(V_link)
#                    for col in range(V_link.shape[1]): 
#                        #print np.asmatrix(V_link[:,col])
#                        #print V_link_t.translation[0:2]
#                        V_link_t.translation = np.asmatrix(V_link[:,col]).T
#                        V_link_frame[:,col] = (self.r.data.oMi[i+1].inverse()*V_link_t).translation.T  
#                    '''                        
##                    for pt in range(V_link.shape[1]): 
##                        #print V_link
##                        ps = self.r.data.oMi[i+1].act(V_link[:,pt])
##                        #ps = self.r.data.oMi[i+1].act(np.hstack((V_link[:,pt],0)))
##                        p = skew(ps)
##                        V_LINK_JC = self.r.frameJacobian(self.q,fid)[0:3,:]- p*self.r.frameJacobian(self.q,fid)[3:6,:]
##                        V_LINK_VEL[:,pt] = np.dot(V_LINK_JC,self.v)                             
##                    #frameVel = self.r.frameVelocity(fid)
##                    #ADD##V_link =  np.vstack((V_link,np.zeros((1,V_link.shape[1]))))
##                    #V_link_vel = frameVel.linear + skew(frameVel.angular)*V_link
##                    V_link_vel = V_LINK_VEL
#                    ##NEW##   
#                except:
#                    vadd = np.ones((V_link.shape[0],V_link.shape[1]))*1e3
#                    V_link += vadd
#                    print 'Numerical Inconsistency'
#                    return self.vcom,self.vdcom,self.vcom_v,self.vdcom1
#                break        
#            vlinknew = np.zeros((2,self.B_sp.shape[0]))
#            vlinknew_vel = np.zeros((2,self.B_sp.shape[0]))
#            for k in range(self.B_sp.shape[0]):
#                ### com ###
#                # find out vertex that minimizes the dot product of each face with the vertices.
#                index_com = np.argmin(np.dot(self.B_sp[k,:],V_link[0:2,:]))
#                # update the newly approximated polytope
#                ##NEW##
#                omega   = np.sqrt(9.81/self.x_com[2,0]);
#                V_link_vel_c = (self.dt+1/omega)*V_link_vel                
#                index_com_v = np.argmin(np.dot(self.B_sp[k,:],V_link_vel_c[0:2,:]))                         
#                vlinknew_vel[:,k] = np.copy(np.matrix(V_link_vel[0:2,index_com_v])).T 
#                ##NEW##                
#                vlinknew[:,k] = np.copy(np.matrix(V_link[0:2,index_com]))                  
#            vlinks_sum += vlinknew 
#            vlinks_vel_sum += vlinknew_vel
#            
#            vlever = self.r.data.oMi[i+1].act(self.r.model.inertias[i+1].lever)*self.r.model.inertias[i+1].mass
#            Aln,bln = compute_convex_hull(vlinknew)
#            d,res1 = check_point_polytope(Aln,bln,vlever[0:2]) 
#            totalmass += self.r.model.inertias[i+1].mass
#
#        vcomc = (1/totalmass)*vlinks_sum
#        vcomc_vel = (1/totalmass)*vlinks_vel_sum 
#        #print '---'
#        #print self.dx_com
#        #print vcomc_vel
#        #print '~~~'
#        Aln,bln = compute_convex_hull(vcomc)        
#        vdcomc = np.matlib.zeros((vcomc.shape[0],vcomc.shape[1]))      
#        vdcomc_1 = np.matlib.zeros((vcomc.shape[0],vcomc.shape[1]))   
#        for k in range(vcomc.shape[1]):
#            comxy = np.copy(vcomc[:,k])
#            comvelxy = np.copy(vcomc_vel[:,k])
#            vdcomc_1[:,k] =comxy + comvelxy[0:2]/np.sqrt(9.81/self.x_com[2])
#            vdcomc[:,k] =  comxy + self.dx_com[0:2]/np.sqrt(9.81/self.x_com[2]) 
##        print '---'
##        print vdcomc
##        print vdcomc_1
##        print '~~~' 
#        '''  
#        d,res1 = check_point_polytope(Aln,bln,self.com_pinocchio[0:2]) 
#        if res1 == False:
#            print 'The global com polytope doesnt contain the nominal com '
#            for m in range(d.shape[1]):
#                 print d[0,m]
#            #(ax,l) =  plot_polytope(Aln, bln, V=None,color='ordered',ax=None,lw=2,dots=False)  
#            #ax.scatter(self.compinocchio[0,0],self.compinocchio[1,0],400,color=COLOR[34],marker='o',label=str(4))              
#        '''
#        return vcomc,vdcomc,vcomc_vel,vdcomc_1              

    #WORK NEED TO BE DONE#    
    def computeGlobalCOMPolytopeModified(self,V,N):
        # initialize params
        joint_ns = 31     
        vlinks_sum= np.matlib.zeros((2,self.B_sp.shape[0]))
        vlinks_vel_sum= np.matlib.zeros((2,self.B_sp.shape[0]))  
        # works only when map_index shape equals the number of polytope
        totalmass = 0
        for i in range(joint_ns):
            idx = self.r.model.names[i+1]
            fid = self.getFrameId(idx)  
            frameVel = self.r.frameVelocity(fid)
            frameJacobian = self.r.frameJacobian(self.q,fid,False,False)
            nommass = self.r.model.inertias[i+1].mass
            a = int(np.sum(N[0,0:i]))
            b = int(N[0,i]+a) 
            vlink = np.matlib.zeros((3,int(N[0,i])))
            vlink_vel = np.matlib.zeros((3,vlink.shape[1])) 
            m = 0
            for j in range(a,b):
                vlink[:,m] = self.r.data.oMi[i+1].act(V[:,j])                
                m += 1
            while True:
                try:
                    Av,bv = compute_convex_hull(vlink)
                    vlink = poly_face_to_span(-Av,bv) 
                    #vlink[2,:] = np.matlib.zeros((1,vlink.shape[1]))
                    #vlink_vel = frameVel.linear + np.dot(skew(frameVel.angular),vlink)
                    for pt in range(vlink.shape[1]): 
                        #p = skew(self.r.data.oMi[i+1].act(vlink[:,pt]))
                        #vlink[2,pt] = 0
                        #p = skew(vlink[:,pt])
                        #vlink_jc = self.r.frameJacobian(self.q,fid)[0:3,:]- p*self.r.frameJacobian(self.q,fid)[3:6,:]
                        #vlink_vel[:,pt] = np.dot(vlink_jc,self.v) 
                        #print 'it'
                        #print vlink[:,pt].T
                        #print np.dot(skew(frameVel.angular),vlink[:,pt].T)
                        skew_vlink = skew(np.asmatrix(vlink[:,pt]).T)
                        jp = frameJacobian[0:3,:]- skew_vlink*frameJacobian[3:6,:]
                        vlink_vel[:,pt] = jp * self.v
                        #vlink_vel[:,pt] = frameVel.linear + np.dot(skew(frameVel.angular),np.asmatrix(vlink[:,pt]).T)
                        #vlink_vel = frameVel.linear + np.dot(skew(frameVel.angular),vlink[:,pt])
                        #print pt
                    #print vlink_vel
                except:
                    r = np.ones((vlink.shape[0],vlink.shape[1]))*1e-1;
                    vlink = vlink +r    
                    print '---Exception----'
                    print i
                    print '----------------'
                    return self.vcom,self.vdcom,self.vcom_v,self.vdcom1
                    continue     
                break       
            # multiply with mass
            mass_links_bound = [self.r.model.inertias[i+1].mass-(self.r.model.inertias[i+1].mass*self.MAX_MASS_ERROR),self.r.model.inertias[i+1].mass+(self.r.model.inertias[i+1].mass*self.MAX_MASS_ERROR)]  
            V_mlb = mass_links_bound[0]*vlink
            V_mhb = mass_links_bound[1]*vlink
            V_vel_mlb = mass_links_bound[0]*vlink_vel
            V_vel_mhb = mass_links_bound[1]*vlink_vel            
            V_link = np.hstack((V_mhb,V_mlb))
            V_link_vel =  np.hstack((V_vel_mhb,V_vel_mlb))     
            vlinknew = np.zeros((2,self.B_sp.shape[0]))          
            omega   = np.sqrt(9.81/self.x_com[2,0]);
            vlinknew_vel = np.zeros((2,self.B_sp.shape[0]))
            '''
            angle_res = np.deg2rad(360/20);
            coml = self.r.data.oMi[i+1].act(self.r.model.inertias[i+1].lever)
            scoml = skew(coml)
            jp = frameJacobian[0:3,:]- scoml*frameJacobian[3:6,:]
            vlinknew_vel_2 = jp * self.v
            vlinknew_vel_2 = nommass *  vlinknew_vel_2
            '''
            V_link_vel_c = (self.dt+1/omega)*V_link_vel
            for k in range(self.B_sp.shape[0]):
                ### com ###      
                #Bst_sp = np.matrix((np.cos(k*angle_res),np.sin(k*angle_res)))
                #index_com_v = np.argmin(np.dot(Bst_sp,V_link_vel[0:2,:])) 
                #index_com = np.argmin(np.dot(Bst_sp,V_link[0:2,:]))
                index_com_v = np.argmin(np.dot(self.B_sp[k,:],V_link_vel_c[0:2,:])) 
                index_com = np.argmin(np.dot(self.B_sp[k,:],V_link[0:2,:]))
                vlinknew[:,k] = np.copy(np.matrix(V_link[0:2,index_com]))                                        
                vlinknew_vel[:,k] = np.copy(np.matrix(V_link_vel[0:2,index_com_v])).T                  
            vlinks_sum += vlinknew
            vlinks_vel_sum += vlinknew_vel
            #vlinks_vel_sum_2 += vlinknew_vel_2  
            vlever = self.r.data.oMi[i+1].act(self.r.model.inertias[i+1].lever)*self.r.model.inertias[i+1].mass
            Aln,bln = compute_convex_hull(vlinknew)
            d,res1 = check_point_polytope(Aln,bln,vlever[0:2]) 
            totalmass += self.r.model.inertias[i+1].mass
        vcomc = (1/totalmass)*vlinks_sum
        vcomc_vel = (1/totalmass)*vlinks_vel_sum
        Aln,bln = compute_convex_hull(vcomc)        
        vdcomc = np.matlib.zeros((vcomc.shape[0],vcomc.shape[1]))      
        vdcomc_1 = np.matlib.zeros((vcomc.shape[0],vcomc.shape[1])) 
        for k in range(vcomc.shape[1]):
            comxy = np.copy(vcomc[:,k])
            comvelxy = np.copy(vcomc_vel[:,k])
            vdcomc_1[:,k] =comxy +comvelxy[0:2]/np.sqrt(9.81/self.x_com[2])
            vdcomc[:,k] =  comxy + self.dx_com[0:2]/np.sqrt(9.81/self.x_com[2]) 
        return vcomc,vdcomc,vcomc_vel,vdcomc_1    
      
    def updateGlobalCOMPolytope(self):
        #q = toPinocchio(self.q)
        #self.com_pinocchio = se3.centerOfMass(self.r.model,self.r.data,self.q,True)
        self.vcom,self.vdcom,self.vcom_v,self.vdcom1 = self.computeGlobalCOMPolytopeModified(self.V,self.N)
                
    def getFrameId(self, frameName):
        if(self.r.model.existFrame(frameName)==False):
            raise NameError("[InvDynFormUtil] ERROR: frame %s does not exist!"%frameName);
        return self.r.model.getFrameId(frameName);

    ''' ********** ENABLE OR DISABLE CONTACT CONSTRAINTS ********** '''

    def removeUnilateralContactConstraint(self, constr_name):
        found = False;
        for i in range(len(self.rigidContactConstraints)):
            if(self.rigidContactConstraints[i].name==constr_name):
                del self.rigidContactConstraints[i];
                del self.rigidContactConstraints_p[i];
                del self.rigidContactConstraints_N[i];
                del self.rigidContactConstraints_fMin[i];
                del self.rigidContactConstraints_mu[i];
                found = True;
                break;
        if(found==False):
            for i in range(len(self.bilateralContactConstraints)):
                if(self.bilateralContactConstraints[i].name==constr_name):
                    del self.bilateralContactConstraints[i];
                    found=True;
                    break;
            if(found==False):
                raise ValueError("[InvDynForm] ERROR: constraint %s cannot be removed because it does not exist!" % constr_name);
        self.updateInequalityData();
        
        
    def addUnilateralContactConstraint(self, constr, contact_points, contact_normals, fMin, mu):
        self.rigidContactConstraints        += [constr];
        self.rigidContactConstraints_p      += [contact_points];
        self.rigidContactConstraints_N      += [contact_normals];
        self.rigidContactConstraints_fMin   += [fMin];
        self.rigidContactConstraints_mu     += [mu];
        self.updateInequalityData();
        
    def existUnilateralContactConstraint(self, constr_name):
        res = [c.name for c in self.rigidContactConstraints if c.name==constr_name];
        return True if len(res)>0 else False;
        
    def addTask(self, task, weight):
        self.tasks        += [task];
        self.task_weights += [weight];
        
    def removeTask(self, task_name):
        for (i,t) in enumerate(self.tasks):
            if t.name==task_name:
                del self.tasks[i];
                del self.task_weights[i];
                return True;
        raise ValueError("[InvDynForm] ERROR: task %s cannot be removed because it does not exist!" % task_name);

    def setTaskWeight(self, task_name, weight):
        for (i,t) in enumerate(self.tasks):
            if t.name==task_name:
                self.task_weights[i] = weight;
                return True;
        raise ValueError("[InvDynForm] ERROR: task %s does not exist!" % task_name);
        
        
    def updateSupportPolygon(self):
        ''' Compute contact points and contact normals in world frame '''
        ncp = int(np.sum([p.shape[1] for p in self.rigidContactConstraints_p]));
        #print self.rigidContactConstraints_p
        self.contact_points  = zeros((3,ncp));
        self.contact_normals = zeros((3,ncp));
        self.contact_points_reduced = zeros((3,ncp));
        mu_s = zeros(ncp);
        i = 0;
        print self.rigidContactConstraints_p;
        for (constr, P, N, mu) in zip(self.rigidContactConstraints, self.rigidContactConstraints_p, 
                                      self.rigidContactConstraints_N, self.rigidContactConstraints_mu):
            oMi = self.r.framePosition(constr._frame_id);
            for j in range(P.shape[1]):
                self.contact_points[:,i]  = oMi.act(P[:,j]);
                self.contact_normals[:,i] = oMi.rotation * N[:,j];
                mu_s[i,0] = mu[0];
                i += 1;      
                
        if(ncp==0 or not self.COMPUTE_SUPPORT_POLYGON):
            self.B_sp = zeros((0,2));
            self.b_sp = zeros(0);
        else:
            avg_z = np.mean(self.contact_points[2,:]);
            if(np.max(np.abs(self.contact_points[2,:] - avg_z)) < 1e-3):
                ''' Contact points are coplanar so I can simply compute the convex hull of 
                    vertical projection of contact points'''   
                (self.B_sp, self.b_sp) = compute_convex_hull(self.contact_points[:2,:].A);
            else:
                (H,h) = compute_GIWC(self.contact_points.T, self.contact_normals.T, mu_s);
                (self.B_sp, self.b_sp) = compute_support_polygon(H, h, self.M[0,0], np.array([0.,0.,-9.81]), eliminate_redundancies=True);
                self.B_sp *= -1.0; 
            # normalize inequalities
            for i in range(self.B_sp.shape[0]):
                tmp = np.linalg.norm(self.B_sp[i,:]);
                if(tmp>1e-6):
                    self.B_sp[i,:] /= tmp;
                    self.b_sp[i]   /= tmp;

            # add a margin in the support polygon for safety reason
            
            self.b_sp -= self.POLYTOPE_MARGIN
            print self.b_sp
#            self.plotSupportPolygon();
            self.B_sp = np.matrix(self.B_sp);
            self.b_sp = np.matrix(self.b_sp).T;  
            self.contact_points_reduced = poly_face_to_span(-np.asarray(self.B_sp),np.asarray(self.b_sp))
        self.support_polygon_computed = True;
            
    ''' Get the matrix B and vector b representing the 2d support polygon as B*x+b>=0 '''
    def getSupportPolygon(self):
        if(not self.support_polygon_computed):
            self.updateSupportPolygon();
        return (np.matrix.copy(self.B_sp), np.matrix.copy(self.b_sp));
        
    def plotSupportPolygon(self):
        import matplotlib.pyplot as plt
        (ax,line) = plot_polytope(-self.B_sp, self.b_sp); 
        ax.scatter(self.x_com[0,0], self.x_com[1,0], c='r', marker='o', s=100);
        for i in range(self.contact_points.shape[1]):
            ax.scatter(self.contact_points[0,i], self.contact_points[1,i], c='k', marker='o', s=100);
        plt.show();
        
        
    ''' ********** ENABLE OR DISABLE INEQUALITY CONSTRAINTS ********** '''
    def enableJointLimits(self, enable=True, IMPOSE_POSITION_BOUNDS=True, IMPOSE_VELOCITY_BOUNDS=True, 
                          IMPOSE_VIABILITY_BOUNDS=True, IMPOSE_ACCELERATION_BOUNDS=True):
        self.ENABLE_JOINT_LIMITS = enable;
        self.IMPOSE_POSITION_BOUNDS = IMPOSE_POSITION_BOUNDS;
        self.IMPOSE_VELOCITY_BOUNDS = IMPOSE_VELOCITY_BOUNDS;
        self.IMPOSE_VIABILITY_BOUNDS = IMPOSE_VIABILITY_BOUNDS;
        self.IMPOSE_ACCELERATION_BOUNDS = IMPOSE_ACCELERATION_BOUNDS;
        self.updateInequalityData();
        
    def enableTorqueLimits(self, enable=True):
        self.ENABLE_TORQUE_LIMITS = enable;
        self.updateInequalityData();
        
    def enableForceLimits(self, enable=True):
        self.ENABLE_FORCE_LIMITS = enable;
        self.updateInequalityData();
        
    def enableCapturePointLimits(self, enable=True):
        self.ENABLE_CAPTURE_POINT_LIMITS = enable;
        self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST = not(enable);
        self.updateInequalityData();

    def enableCapturePointLimitsRobust(self, enable=True,enableVel=False):
        self.INCLUDE_VEL_UNCERTAINTIES = enableVel;
        self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST = enable;
        self.ENABLE_CAPTURE_POINT_LIMITS = not(enable);
        self.MAX_MASS_ERROR = self.inertiaError[0]
        self.MAX_COM_ERROR = self.inertiaError[1]
        self.MAX_INERTIA_ERROR = self.inertiaError[2]           
        self.updateInequalityData();
      
   
    ''' ********** SET ROBOT STATE ********** '''
    def setPositions(self, q, updateConstraintReference=True):
        self.q = np.matrix.copy(q);
        
        if(updateConstraintReference):
            if(self.USE_JOINT_VELOCITY_ESTIMATOR):
                raise Exception("Joint velocity estimator not implemented yet");
                self.estimator.init(self.dt,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,self.JOINT_VEL_ESTIMATOR_DELAY,True);
                self.baseVelocityFilter = FirstOrderLowPassFilter(self.dt, self.BASE_VEL_FILTER_CUT_FREQ , zeros(6));
            self.r.forwardKinematics(q);
            for c in self.rigidContactConstraints:
                Mref = self.r.position(q, c._link_id, update_geometry=False);
                c.refTrajectory.setReference(Mref);
#                dx = np.dot(c.task.jacobian.value, self.dq);
#                if(np.linalg.norm(dx)>EPS):
#                    print "[InvDynForm] Contact constraint velocity: %.3f" % np.linalg.norm(dx);
            for c in self.bilateralContactConstraints:
                Mref = self.r.position(q, c._link_id, update_geometry=False);
                c.refTrajectory.setReference(Mref);
            self.updateSupportPolygon();
            self.firstTime = True
            self.dJ_com = zeros((3,self.n+6)); 
            
        return self.q;
    
    def setVelocities(self, v):
        if(self.USE_JOINT_VELOCITY_ESTIMATOR):
            raise Exception("Joint velocity estimator not implemented yet");
        else:
            self.v = np.matrix.copy(v);
        return self.v;
        
    def setNewSensorData(self, t, q, v):
        self.t = t;
        self.setPositions(q, updateConstraintReference=False);
        self.setVelocities(v);
        
        self.r.computeAllTerms(q, v);
        self.r.framesKinematics(q);
        self.x_com    = self.r.com(q, update_kinematics=True);

        if(self.firstTime==False):
            self.dJ_com   = (self.r.Jcom(q, update_kinematics=False) - self.J_com)/self.dt;  
        self.J_com    = self.r.Jcom(q, update_kinematics=False);    
        

          
        self.dd_com        = np.dot(self.dJ_com, self.v);
        self.M        = self.r.mass(q, update_kinematics=False);
        self.Ag       = self.r.momentumJacobian(q, v);
        if(self.ACCOUNT_FOR_ROTOR_INERTIAS):
            if(self.freeFlyer):
                self.M[6:,6:]   += self.Md;
            else:
                self.M   += self.Md;
        self.h        = self.r.bias(q,v, update_kinematics=False);
#        self.h          += self.JOINT_FRICTION_COMPENSATION_PERCENTAGE*np.dot(np.array(JOINT_VISCOUS_FRICTION), self.v);
        self.dx_com     = np.dot(self.J_com, self.v);


         # CHECK WITH JUSTIN
        '''
        if self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST == True:
            self.x_com_modified = self.r_modified.com(q, update_kinematics=True)       
            self.J_com_modified    = self.r_modified.Jcom(q, update_kinematics=False);
            self.dx_com_modified     = np.dot(self.J_com_modified, self.v);        
        '''    
        com_z           = self.x_com[2]; #-np.mean(self.contact_points[:,2]);
        if(com_z>0.0):
            self.cp         = self.x_com[:2] + self.dx_com[:2]/np.sqrt(9.81/com_z);
            #if self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST == True:
            #    self.cp_modified = self.x_com_modified[:2] + self.dx_com[:2]/np.sqrt(9.81/com_z);
        else:
            self.cp = zeros(2);
            #self.cp_modified = zeros(2);
        #if self.V != None and self.B_sp != np.zeros((0,2)) and self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST == True:
        #    self.updateGlobalCOMPolytope()     
        self.updateConstrainedDynamics();
        self.firstTime = False;
        

    def updateConstrainedDynamics(self):
        t = self.t;
        q = self.q;
        v = self.v;
        k = self.k;
        nv = self.nv;
        i = 0;
        for constr in self.rigidContactConstraints:
            dim = constr.dim
            (self.Jc[i:i+dim,:], self.dJc_v[i:i+dim], self.ddx_c_des[i:i+dim]) = constr.dyn_value(t, q, v, local_frame=False);
            i += dim;
        for constr in self.bilateralContactConstraints:
            dim = constr.dim
            (self.Jc[i:i+dim,:], self.dJc_v[i:i+dim], self.ddx_c_des[i:i+dim]) = constr.dyn_value(t, q, v, local_frame=False);
            i += dim;
        self.Minv        = np.linalg.inv(self.M);
        if(self.k>0):
            self.Jc_Minv     = np.dot(self.Jc, self.Minv);
            self.Lambda_c    = np.linalg.inv(np.dot(self.Jc_Minv, self.Jc.T) + 1e-10*np.matlib.eye(self.k));
            self.Jc_T_pinv   = np.dot(self.Lambda_c, self.Jc_Minv);
            self.Nc_T        = np.matlib.eye(self.nv) - np.dot(self.Jc.T, self.Jc_T_pinv);
            self.dx_c        = np.dot(self.Jc, self.v);
        else:
            self.Nc_T        = np.matlib.eye(self.nv);
        
        # Compute C and c such that y = C*tau + c, where y = [dv, f, tau]
        self.C[0:nv,:]      = np.dot(self.Minv, np.dot(self.Nc_T, self.S_T));
        self.C[nv:nv+k,:]   = -np.dot(self.Jc_T_pinv, self.S_T);
        self.C[nv+k:,:]     = np.matlib.eye(self.na);
        self.c[0:nv]        = - np.dot(self.Minv, (np.dot(self.Nc_T,self.h) + np.dot(self.Jc.T, np.dot(self.Lambda_c, self.dJc_v - self.ddx_c_des))));
        self.c[nv:nv+k]     = np.dot(self.Lambda_c, (np.dot(self.Jc_Minv, self.h) - self.dJc_v + self.ddx_c_des));

        
    def computeCostFunction(self, t):
        n_tasks = len(self.tasks);
        dims    = np.empty(n_tasks, np.int);
        J       = n_tasks*[None,];
        drift   = n_tasks*[None,];
        a_des   = n_tasks*[None,];
        dim = 0;
        for k in range(n_tasks):
            J[k], drift[k], a_des[k] = self.tasks[k].dyn_value(t, self.q, self.v);
            dims[k] = a_des[k].shape[0];
            dim += dims[k];
        A = zeros((dim, self.nv+self.k+self.na));
        a = zeros(dim);
        i = 0;
        for k in range(n_tasks):
            A[i:i+dims[k],:self.nv] = self.task_weights[k]*J[k];
            a[i:i+dims[k]]          = self.task_weights[k]*(a_des[k] - drift[k]);
            i += dims[k];
        D       = np.dot(A,self.C);
        d       = a - np.dot(A,self.c);
        return (D,d);
    
    
    ''' ********** GET ROBOT STATE ********** '''        
    def getAngularMomentum(self):
        return (self.Ag * self.v)[3:,0];
#        I = self.M[3:6,3:6];
#        return np.dot(np.linalg.inv(I), np.dot(self.M[3:6,:], self.v));
        
    def getZmp(self, f_l, f_r):
        return zeros(2);
#        self.x_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,3];  # position right foot
#        self.x_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,3];  # position left foot
#        self.R_rf = np.array(self.constr_rfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix right foot
#        self.R_lf = np.array(self.constr_lfoot.opPointModif.position.value)[0:3,0:3];  # rotation matrix left foot
#        
#        self.zmp_l = zeros(3);
#        self.zmp_r = zeros(3);
#        if(abs(f_l[2])>1e-6):
#            self.zmp_l[0] = -f_l[4]/f_l[2];
#            self.zmp_l[1] = f_l[3]/f_l[2];
#        self.zmp_l = self.x_lf + np.dot(self.R_lf, self.zmp_l);
#        if(abs(f_r[2])>1e-6):
#            self.zmp_r[0] = -f_r[4]/f_r[2];
#            self.zmp_r[1] = f_r[3]/f_r[2];
#        self.zmp_r = self.x_rf + np.dot(self.R_rf, self.zmp_r);
#        self.zmp = (f_l[2]*self.zmp_l[:2] + f_r[2]*self.zmp_r[:2]) / (f_l[2]+f_r[2]);
#        return np.matrix.copy(self.zmp);

    
   
    ''' ********** CREATE INEQUALITY CONSTRAINTS ********** '''

    ''' Computes a matrix B and a vector b such that the inequalities:
            B*dv + b >= 0
        ensures that the capture point at the next time step will lie
        inside the support polygon. Note that the vector dv contains the
        accelerations of base+joints of the robot. This methods assumes that
        x_com, dx_com, J_com, B_sp and b_sp have been already computed.
    '''
    def createCapturePointInequalities(self, footSizes = None):    
        dt      = self.dt;
        omega   = np.sqrt(9.81/self.x_com[2])[0,0];
        x_com   = self.x_com[0:2];  # only x and y coordinates
        dx_com  = self.dx_com[0:2];
        B    = (0.5*dt*dt + dt/omega)*np.dot(self.B_sp, self.J_com[0:2,:]);
        b    = self.b_sp + np.dot(self.B_sp, x_com + (dt+1/omega)*dx_com + ((0.5*dt*dt + dt/omega)*self.dd_com[0:2]));
        return (B,b);
        
    def createCapturePointInequalitiesRobust(self, footSizes = None):  
        self.updateGlobalCOMPolytope() 
        dt      = self.dt;
        omega   = np.sqrt(9.81/self.x_com[2])[0,0];
        x_com   = self.x_com[0:2];  # only x and y coordinates
        dx_com  = self.dx_com[0:2];
        B    = (0.5*dt*dt + dt/omega)*np.dot(self.B_sp, self.J_com[0:2,:]);
        # B_sp are the normals defining the directions of interest in a support polygon.
        # x_com and dx_com should correspond to maximum regions points of the the capture polygon
        c    = np.copy(self.b_sp);
        
        '''
        angle_res = np.deg2rad(20/16); 
        for i in range(16):
                ### com ###      
                Bst_sp = np.matrix((np.cos(i*angle_res),np.sin(i*angle_res)))
                
                c[i,0] = c[i,0] + np.dot(Bst_sp,self.vcom[:,i]+ ((dt+1/omega)*self.vcom_v[:,i]+((0.5*dt*dt + dt/omega)*self.dd_com[0:2])).T.T);         
        '''
        for i in range(self.B_sp.shape[0]):
           #c[i,0] = c[i,0] + np.dot(self.B_sp[i,:],self.vcom[:,i]+ ((dt+1/omega)*dx_com+((0.5*dt*dt + dt/omega)*self.dd_com[0:2])).T.T);  
           if self.INCLUDE_VEL_UNCERTAINTIES == True:
               c[i,0] = c[i,0] + np.dot(self.B_sp[i,:],self.vcom[:,i]+ ((dt+1/omega)*self.vcom_v[:,i]+((0.5*dt*dt + dt/omega)*self.dd_com[0:2])).T.T);         
           else:
               c[i,0] = c[i,0] + np.dot(self.B_sp[i,:],self.vcom[:,i]+ ((dt+1/omega)*dx_com+((0.5*dt*dt + dt/omega)*self.dd_com[0:2])).T.T); 
        return (B,c);         

    def createJointAccInequalitiesViability(self):
        n  = self.na;
        B  = zeros((2*n,n));
        b  = zeros(2*n);
                
        B[:n,:]  =  np.matlib.identity(n);
        B[n:,:]  = -np.matlib.identity(n);

        # Take the most conservative limit for each joint
        dt = max(self.JOINT_POS_PREVIEW,self.JOINT_VEL_PREVIEW)*self.dt;
        self.ddqMax[:,0]  = self.MAX_JOINT_ACC;
        self.ddqStop[:,0] = self.MAX_MIN_JOINT_ACC;
        (ddqLB, ddqUB) = computeAccLimits(self.q[7:], self.v[6:], self.qMin[7:], self.qMax[7:], self.dqMax[6:], self.ddqMax, 
                                          dt, False, self.ddqStop, self.IMPOSE_POSITION_BOUNDS, self.IMPOSE_VELOCITY_BOUNDS, 
                                          self.IMPOSE_VIABILITY_BOUNDS, self.IMPOSE_ACCELERATION_BOUNDS);
        self.ddqMinFinal = ddqLB;
        self.ddqMaxFinal = ddqUB;
        
        b[:n]    = -self.ddqMinFinal;
        b[n:]    = self.ddqMaxFinal;

        if(np.isnan(b).any()):
            print " ****** ERROR ***** Joint acceleration limits contain nan";
        
        return (B,b);
    
    
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
        
        
    ''' Compute the matrix A and the vectors lbA, ubA such that:
            lbA <= A*tau <= ubA
        ensures that all the inequality constraints the system is subject to are satisfied.
        Before calling this method you should call setNewSensorData to set the current state of 
        the robot.
    '''
    def createInequalityConstraints(self):
        n = self.na;

        if(self.ENABLE_JOINT_LIMITS):
            (B_q, b_q) = self.createJointAccInequalitiesViability();
            self.B[self.ind_acc_in, 6:n+6]      = B_q;
            self.b[self.ind_acc_in]             = b_q;
        #print 'What the problem?'
        #print self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST   
        if(self.ENABLE_CAPTURE_POINT_LIMITS_ROBUST):
            (B_cp, b_cp) = self.createCapturePointInequalitiesRobust();
            self.B[self.ind_cp_in, :n+6]        = B_cp;
            self.b[self.ind_cp_in]              = b_cp;            
        elif(self.ENABLE_CAPTURE_POINT_LIMITS):
            (B_cp, b_cp) = self.createCapturePointInequalities();            
            self.B[self.ind_cp_in, :n+6]        = B_cp;
            self.b[self.ind_cp_in]              = b_cp;
        
        self.G       = np.dot(self.B, self.C);
        self.glb     = self.b + np.dot(self.B, self.c);
        self.gub     = 1e10*np.matlib.ones((self.m_in,1))
        return (self.G, -self.glb, self.gub, self.lb, self.ub);
    
        
    def createForceRegularizationTask(self, w_f):
        n = self.n;      # number of joints
        k = self.k;
        A = zeros((12,2*n+6+k));
        A[:,n+6:n+6+12]  = np.diag(w_f);
        D       = np.dot(A,self.C);
        d       = - np.dot(A,self.c);
        return (D,d);
