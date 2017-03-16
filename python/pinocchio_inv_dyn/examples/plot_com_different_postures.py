# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 22:59:39 2017

@author: nemogiftsun
"""
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import example_hrp2_config_ng as conf
import numpy as np
import pinocchio as se3
from pickle import load
import pinocchio_inv_dyn.viewer_utils as viewer_utils
viewer_utils.ENABLE_VIEWER = True
from pinocchio.utils import zero as zeros
from pinocchio_inv_dyn.add_inertial_uncertainties import generate_new_inertial_params
from pinocchio_inv_dyn.polytope_conversion_utils import poly_face_to_span,poly_span_to_face,cone_span_to_face
from pinocchio_inv_dyn.convex_hull_util import compute_convex_hull
from pinocchio_inv_dyn.geom_utils import plot_polytope
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime



def compute_support_polygon(r,contacts):
    rigidContactConstraints_p = []
    rigidNames_p = []
    for (name, PN) in contacts.iteritems():
        Pi = np.matrix(PN['P']).T;
        rigidContactConstraints_p += [Pi]; 
        rigidNames_p += [name]
    ncp = int(np.sum([p.shape[1] for p in rigidContactConstraints_p]));
    contact_points  = zeros((3,ncp));
    i = 0
    for P,name in zip(rigidContactConstraints_p,rigidNames_p):
        print name
        print P
        oMi = r.framePosition(r.model.getFrameId(name));
        for j in range(P.shape[1]):
            contact_points[:,i]  = oMi.act(P[:,j]);
            i += 1; 
        (B_sp,b_sp) = compute_convex_hull(contact_points[0:2,:].A);     
        # normalize inequalities
        for i in range(B_sp.shape[0]):
            tmp = np.linalg.norm(B_sp[i,:]);
            if(tmp>1e-6):
                B_sp[i,:] /= tmp;
                b_sp[i]   /= tmp;
        #b_sp -= 0.005
        B_sp = np.matrix(B_sp);
        b_sp = np.matrix(b_sp).T;  
        cpr = poly_face_to_span(-np.asarray(B_sp),np.asarray(b_sp))     
    return (B_sp,b_sp,cpr)


cp = np.matrix([[ 0.26,  0.26,  0.08,  0.08, -0.06, -0.06, -0.24, -0.24],
               [ 0.15,  0.04,  0.04,  0.15, -0.04, -0.15, -0.15, -0.04],
               [-0.  , -0.  , -0.  , -0.  , -0.01, -0.01, -0.01, -0.01]])


def area(p):
    return 0.5 * abs(sum(np.linalg.norm(np.array(x1)-np.array(x2)) for (x1,x2) in segments(p)))
                             
def segments(p):
    return zip(p, p[1:] + [p[0]])

def computeGlobalCOMPolytopeModified(r,V,N,B_sp,b_sp):
    # initialize params
    joint_ns = 31     
    vlinks_sum= np.matlib.zeros((2,B_sp.shape[0]))
    totalmass = 0
    for i in range(joint_ns):
        a = int(np.sum(N[0,0:i]))
        b = int(N[0,i]+a) 
        vlink = np.matlib.zeros((3,int(N[0,i])))
        m = 0
        for j in range(a,b):
            vlink[:,m] = r.data.oMi[i+1].act(V[:,j])                
            m += 1
        Av,bv = compute_convex_hull(vlink)
        vlink = poly_face_to_span(-Av,bv) 
        # multiply with mass
        mass_links_bound = [r.model.inertias[i+1].mass-(r.model.inertias[i+1].mass*conf.MAX_MASS_ERROR),r.model.inertias[i+1].mass+(r.model.inertias[i+1].mass*conf.MAX_MASS_ERROR)]  
        V_mlb = mass_links_bound[0]*vlink
        V_mhb = mass_links_bound[1]*vlink         
        V_link = np.hstack((V_mhb,V_mlb))
    
        vlinknew = np.zeros((2,B_sp.shape[0])) 
        for k in range(B_sp.shape[0]):
            index_com = np.argmin(np.dot(B_sp[k,:],V_link[0:2,:]))
            vlinknew[:,k] = np.copy(V_link[0:2,index_com]).T                       
        vlinks_sum += vlinknew
        totalmass += r.model.inertias[i+1].mass
    vcomc = (1/totalmass)*vlinks_sum
    Ac,bc = compute_convex_hull(vcomc)
    vcomc = poly_face_to_span(-Ac,bc)
    return vcomc  


robotwrapper  = RobotWrapper(conf.urdfFileName, conf.model_path, se3.JointModelFreeFlyer())
viewer1 = viewer_utils.Viewer('viewer',robotwrapper,'robot1');

f = open("../../../data/configs_statically_balanced_fixed_contacts","r")
qs=load(f)
f.close()
qsmat = np.matrix(qs)
V,N,error_model = generate_new_inertial_params(conf.MAX_MASS_ERROR,conf.MAX_COM_ERROR,conf.MAX_INERTIA_ERROR)
viewer1.updateRobotConfig(qsmat[12,:])
#robotwrapper.forwardKinematics(qsmat[12,:])
(Bsp,bsp,cpr)=compute_support_polygon(robotwrapper,contacts)
print bsp
#print cpr
#viewer1.addPolytope('sp_controller',cpr,robotName='robot1',color = [0,0.7,0.4,0.8])     
Ns = []      
#viewer1.addSphere('sphere', 0.002, np.zeros(3), color=(1.0, 1.0, 0, 1))
SAMPLES = 10
d = np.zeros((4,SAMPLES));
com = np.zeros((3,SAMPLES))
for n in range(SAMPLES):
    time.sleep(0.1)
    viewer1.updateRobotConfig(qsmat[n,:])
    sphere_name = 'sphere_'+str(n)
    line_name = 'line_'+str(n)
    if n%1000==0:
        print 'Executing  '+str(n)+'th iteration'
    while(True):
        try:
            vc = computeGlobalCOMPolytopeModified(robotwrapper,V,N,Bsp,bsp)
            #viewer1.addPolytope('capture point polytope',vc,robotName='robot1',color = [1,0.8,1,1]) 
            #print 
            #print np.array(np.copy(robotwrapper.com(qsmat[n,:]))).T[0]
            
            com[:,n] =  np.array(np.copy(robotwrapper.com(qsmat[n,:]))).T[0]
            com[2,n] = -0.02
            p= np.copy(com);
            p[2]=com[2,0];
            #d = np.zeros(vc.shape[1])
            #l = []
            for nv in range(vc.shape[1]):
                #point = tuple(vc[0:2,nv])
                #print point
                #l.append(point)
                #p[2]=com[2,0];
                #print 'coms'
                #print vc[0:2,nv] - np.array(com[0:2,:]).T
                d[nv,n] = np.linalg.norm(vc[0:2,nv] - np.array(com[0:2,n]).T)
                #p[2]=d[nv];
                #viewer1.addLine(line_name+str(nv),com,p,color=(1,0,1,1.0))
            #l.append(l[0])
            #print l
            #p[2] -= area(l)
            #print area(l)
            #viewer1.addSphere(sphere_name, 0.002, com[:,n], color=(1.0, 0, 0, 1))
            #ax.scatter(vc[0],vc[1],s=100,color=(1 .0, 0, 0, 1))
            #print 'norms' +str(d[0])+str(d[1])+str(d[2])+str(d[3])
            #viewer1.addLine(line_name,com,p,color=(1,0,1,1.0))  
        except:
            Ns.append(n)
            print "Appending "+str(n)
            pass
        break

np.savez_compressed('../examples/data_com_configurations_'+datetime.now().strftime('%m%d_%H%M%S'),com = com,d=d);    
#plot_color_gradients(ax)  
#x = np.linspace(0.0, 1.0, SAMPLES)
#ax.scatter(com[0,:],com[1,:],c=d[0,:],cmap="Blues")  
f,(ax1, ax2,ax3,ax4) = plt.subplots(1,4,sharey=True)
(ax1,line) = plot_polytope(np.asarray(Bsp),np.asarray(bsp),ax=ax1,color='g')
(ax2,line) = plot_polytope(np.asarray(Bsp),np.asarray(bsp),ax=ax2,color='g')
(ax3,line) = plot_polytope(np.asarray(Bsp),np.asarray(bsp),ax=ax3,color='g')
(ax4,line) = plot_polytope(np.asarray(Bsp),np.asarray(bsp),ax=ax4,color='g')     
a1 = ax1.scatter(com[0,:],com[1,:],c=d[0,:])
a2 = ax2.scatter(com[0,:],com[1,:],c=d[1,:])
a3 = ax3.scatter(com[0,:],com[1,:],c=d[2,:])
a4 = ax4.scatter(com[0,:],com[1,:],c=d[3,:])
ax1.set_title('Direction 0',fontsize=17)
ax2.set_title('Direction 1',fontsize=17)
ax3.set_title('Direction 2',fontsize=17)
ax4.set_title('Direction 3',fontsize=17)
f.colorbar(a1)
#f.colorbar(a2)
plt.show()