# -*- coding: utf-8 -*-
"""
This script can be used to semi-automatically edit the geometric/dynamic model of HRP-2.
In particular, you can add noise on the inertial parameters of the robot to test the robustness
of a controller to uncertainties in the inertial parameters.
Example of the lines to modify in the wrl files:
    centerOfMass -0.01435142 0.00000012 0.22307695
    mass 15.02949000
    momentsOfInertia [0.16102617 -0.00000026 0.00070696 -0.00000027 0.09528000 -0.00000019 0.00070696 -0.00000020 0.19191383]


Created on Mon Apr  6 16:31:47 2015
@author: ngiftsun
"""
import random
import numpy as np

from pinocchio_inv_dyn.geom_utils import auto_generate_polytopes,check_point_polytope
from pinocchio_inv_dyn.polytope_conversion_utils import poly_face_to_span,poly_span_to_face

import pinocchio_inv_dyn.examples.example_hrp2_config_ng as conf
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import pinocchio as se3

def generate_new_inertial_params(MAX_MASS_ERROR, MAX_COM_ERROR, MAX_INERTIA_ERROR, verb=0,freeflyer = True):
    if freeflyer == True:
        rold = RobotWrapper(conf.urdfFileName, conf.model_path, root_joint=se3.JointModelFreeFlyer());
        rnew = RobotWrapper(conf.urdfFileName, conf.model_path, root_joint=se3.JointModelFreeFlyer());
    else:
        rold =  RobotWrapper(conf.urdfFileName, conf.model_path, None);
        rnew =  RobotWrapper(conf.urdfFileName, conf.model_path, None);
    nd = [0,]*31
    Vt = None
    for i in range(1,rnew.model.nbodies):
        com_error = np.random.uniform(-1,1,3);
        com_error /= np.linalg.norm(com_error);
        com_error *= random.random()*MAX_COM_ERROR
        input_com = rnew.model.inertias[i].lever
        output_com = input_com + np.matrix(com_error).T;
        input_mass  = rnew.model.inertias[i].mass
        w = 1 - MAX_MASS_ERROR + random.random()*2*MAX_MASS_ERROR;
        output_mass = w*input_mass;
        rnew.model.inertias[i].lever = output_com;
        rnew.model.inertias[i].mass = output_mass;
        com_essential = np.array(np.vstack((input_com.T,output_com.T)))
        intxply = np.add([-MAX_COM_ERROR,MAX_COM_ERROR],input_com[0,0])
        intyply = np.add([-MAX_COM_ERROR,MAX_COM_ERROR],input_com[1,0])
        intzply = np.add([-MAX_COM_ERROR,MAX_COM_ERROR],input_com[2,0]) 
        interval = [intxply,intyply,intzply]  
        #print com_essential   
        while True:
                try:
                    A,b,V,intr = auto_generate_polytopes(4,interval,com_essential)
                    Alink,blink = poly_span_to_face(V)
                    V = poly_face_to_span(Alink,blink)
                except:
                    print 'Numerical Inconsistency'
                    continue
                break                               
        #verify if both the points are inside  
        d,res1 = check_point_polytope(-Alink,blink,np.array(input_com))
        d,res2 = check_point_polytope(-Alink,blink,np.array(output_com))
        if res1&res2==False:
            print 'not in'
            print i              
        Vt = V if Vt == None else np.hstack((Vt,V))
        nd[i-1] = V.shape[1]    
        
    return np.asmatrix(Vt),np.asmatrix(nd),rnew 

   
if __name__=='__main__':                         
    # mass com_x com_y com_z inertia_xx inertia_xy inertia_xz inertia_yy inertia_yz inertia_zz
    #new_params,old_params = modify_hrp2_model_adapted('HRP2JRLmainsmall_v10.wrl', 0.03, 0.01, 0.2, verb=1);
    vt,nd,rn = generate_new_inertial_params(0.01, 0.01, 0.01)
