# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:03:07 2017

@author: nemogiftsun
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:36:34 2015

@author: adelpret
"""
import numpy as np
import pinocchio_inv_dyn.plot_utils as plot_utils
from pinocchio_inv_dyn.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
import os
import pickle
import example_hrp2_config_ng as conf
import pinocchio_inv_dyn.viewer_utils as viewer_utils
import pinocchio as se3

from time import sleep
import math



            
            
np.set_printoptions(precision=2, suppress=True);
plot_utils.FIGURE_PATH = '../results/test_reachable/check/';
#plot_utils.FIGURE_PATH = '../results/test_drill/check/';
print 'Analyze data in folder', plot_utils.FIGURE_PATH;

COM_SPHERE_RADIUS = 0.005;
CAPTURE_POINT_SPHERE_RADIUS = 0.005;
COM_SPHERE_COLOR = (1, 0, 0, 1);
CAPTURE_POINT_SPHERE_COLOR = (0, 1, 0, 1);
CAPTURE_POINT_CTR_SPHERE_COLOR = (0, 1, 1, 1);

DATA_FILE_NAME              = 'data';
CONSTRAINT_VIOLATION_FILE_NAME = 'constraint_violations';
SAVE_RESULTS                = False;
PLAY_SALIENT_MOMENTS        = False;
DUMP_IMAGE_FILES            = True;
N_DIR = 0;
if(N_DIR<=0):
    N_DIR = len(os.listdir(plot_utils.FIGURE_PATH));
    N_TESTS=N_DIR
N_SOLVERS = 3;
TEXT_FILE_NAME = 'stats.txt';
dt = 0.002;
MAX_TIME = 6.0-dt;
SALIENT_MOMENT_DURATION = MAX_TIME;
SALIENT_MOMENT_SLOWDOWN_FACTOR = 0.2;
ROBOT_MOVE = np.array((0.5,0.5,-0.01))

#WALL_POSITION = 1.32;
WALL_POSITION = 1.17
WALL_COLOR = (135.0/255, 74.0/255, 13.0/255, 0.95);


line_styles     =["k-", "b--", "r:", "c-", "g-"];
solver_names    = ['Classic','Robust','Robust_vel'];
viewer_utils.ENABLE_VIEWER = True;
time_to_reach        = np.zeros((N_DIR,N_SOLVERS));
task_error          = np.zeros((N_DIR,N_SOLVERS));
time_to_fall        = np.zeros((N_DIR,N_SOLVERS));
tmax                = np.zeros((N_DIR,N_SOLVERS));
falls               = np.zeros((N_DIR,N_SOLVERS));
real_falls               = np.zeros((N_DIR,N_SOLVERS));
reach               = np.zeros((N_DIR,N_SOLVERS));
#i_to_fall           = np.zeros((N_DIR,N_SOLVERS), np.int);
itmax              = np.zeros((N_DIR,N_SOLVERS));
itzerov            = np.zeros((N_DIR,N_SOLVERS));
reach_err           = np.zeros((N_DIR,N_SOLVERS));
scores = np.zeros((N_DIR,N_SOLVERS), np.int);
#rh_task_error = np.zeros((N_DIR,N_SOLVERS,MAX_TIME))


'''Camera view parameters'''
#direction
ud =(0,0,1);
#camera profile 1
eye1 = [0.5,0,0.7]; pos1 = [0.5,-3,0.5];
#pos1 = [0.5, -3.5, 0.5];eye1 = [0.5, 0, 0.7];
#camera front
eye2 = [0,0,0.7]; pos2 = [3.,0,0.5];
#camera profile 2
eye3 = [0.5,0,0.7]; pos3 = [0.5,3,0.5]; 
# diagonal view
eye4 = [0.4,0,0.7];pos4= [2.6,2.6,0.7]
#support polygon
eye5 = [0.0,0,0.7]+ROBOT_MOVE;pos5= [0.5,0.5,-1.0];ud5 =(0,-1.,0);
v = np.array((-0.25,0.07,-0.12))
t = np.array((0.,0.025,-0.0))
eye6 = [0.1,0,0.7];pos6= [0.0,0.0,-1.2];ud6 =(0.05,1,1);


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        pass

if(PLAY_SALIENT_MOMENTS):
    robotwrapper = r = RobotWrapper(conf.urdfFileName, conf.model_path, se3.JointModelFreeFlyer())
    viewer = viewer_utils.Viewer('viewer',robotwrapper,'robot1');
    viewer.CAMERA_FOLLOW_ROBOT = False;
    #viewer.moveCamera([0,0,0], [2,2,2], upwardDirection=(0,0,1))
    viewer.activateMouseCamera();
    viewer.addSphere('target', 0.015, np.array([WALL_POSITION-0.01, -0.23, 0.95])+ROBOT_MOVE, (0,0,0), (1,0,0,1), 'OFF');

i = 0;
info_tail = '';
runs = 0
for dirname in os.listdir(plot_utils.FIGURE_PATH):
    runs+=1
    path = plot_utils.FIGURE_PATH+dirname+'/';
    if(not os.path.exists(path)):
        continue;
    print "Gonna open", path+DATA_FILE_NAME+'.npz';
    data = np.load(path+DATA_FILE_NAME+'.npz'); 
    time_to_fall[i,:] = data['time_to_fall'].reshape(N_SOLVERS);
    time_to_reach[i,:] = data['time_to_reach'].reshape(N_SOLVERS);
    reach[i,:] = (time_to_reach>0).sum(axis=0)
    falls[i,:] = data['falls']
    dt = data['dt']
    vel = data['v']
    rh_task_error = data['rh_task_error']
    #itmax[i,:] = data['itmax']
    
    for s in range(N_SOLVERS):
        itmax[i,s] = data['final_time_step'][s]
        tmax[i,s]  = data['final_time'][s] 
#        if falls[i,s] == 1:
#            tmax[i,s] = time_to_fall[i,s] 
#        else:
#            tmax[i,s] = itmax[i,s]*dt 
        '''
        time_to_reach[i,s] = 0
        task_error[i,s] = 0
        if falls[i,s] == 0:
            for c in range(200,rh_task_error.shape[1]):
                if np.max(np.abs(vel[s]),axis=0)[c] < 5e-3: 
                   itzerov[i,s] = c;
                   task_error[i,s] = rh_task_error[s,itzerov[i,s]]
                   time_to_reach[i,s] = itzerov[i,s] *dt
                   break;
        if time_to_reach[i,s] == 0:
           real_falls[i,s] = 1
        '''
        if falls[i,s] == 0:
            task_error[i,s] = rh_task_error[s,itmax[i,s]]
            '''
            for c in range(200,rh_task_error.shape[1]):
                if np.max(np.abs(vel[s]),axis=0)[c] < 1e-3: 
                   itzerov[i,s] = c;
                   task_error[i,s] = rh_task_error[s,itzerov[i,s]]
                   time_to_reach[i,s] = itzerov[i,s] *dt
                   break;
            '''      
        #time_to_reach[i,s] = max(itzerov[i,s]*dt,time_to_reach[i,s])
        #task_error[i,s] = rh_task_error[s,int(time_to_reach[i,s]/dt)]
                  
       
    #viewer.moveCamera(eye1, pos1, ud)
    sp_points_o = data['v_sp_original']
    sp_points_r = data['v_sp_reduced']
    # prepare viewer
    cp_ctrl_i = data['cp_ctrl']
    cp_i = data['cp']
    com_sim = data['x_com_sim']
    vdcom = data['v_cp_robust']
    #viewer.addPolytope('sp_controller_original',(ROBOT_MOVE[0:2]+sp_points_o.T).T,robotName='robot1',color = [0,0.7,0.4,0.8])

    if(PLAY_SALIENT_MOMENTS):
    #viewer.moveCamera(eye5, pos5, ud5)
    #viewer.moveCamera(eye1, pos1, ud)
    # add legends 
        viewer.addSphere('cp_legend',  COM_SPHERE_RADIUS+0.004, ROBOT_MOVE+v,np.array((0.,-0,0)), COM_SPHERE_COLOR, 'OFF');    
        viewer.addSphere('cp_control_legend', COM_SPHERE_RADIUS+0.004, ROBOT_MOVE+v+t, np.zeros(3), CAPTURE_POINT_CTR_SPHERE_COLOR, 'OFF');   
        viewer.addSphere('sp_legend', COM_SPHERE_RADIUS+0.004, ROBOT_MOVE+v+2*t, np.zeros(3),[0,0.7,0.4,0.8], 'OFF');   
        viewer.addSphere('cp_polytope_legend', COM_SPHERE_RADIUS+0.004,ROBOT_MOVE+v+3*t, np.zeros(3),[1,0.8,0.4,1], 'OFF');    
        viewer.addPolytope('sp_controller_reduced',(ROBOT_MOVE[0:2]+sp_points_r.T).T,robotName='robot1',color = [0,0.7,0.4,0.8])
        viewer.addSphere('cp',  COM_SPHERE_RADIUS-0.001, np.zeros(3), np.zeros(3), COM_SPHERE_COLOR, 'OFF');
        viewer.addSphere('cp_control', COM_SPHERE_RADIUS-0.001, np.zeros(3), np.zeros(3), CAPTURE_POINT_CTR_SPHERE_COLOR, 'OFF');   
        #pa = viewer.path+dirname
        pa = plot_utils.FIGURE_PATH+dirname+'/capture'
        mkdir_p(pa);
        for s in range(N_SOLVERS):
            print 'solver'
            end = tmax[i,s]/dt;
            sleep(2)
            if(DUMP_IMAGE_FILES or tmax[i,s]<=MAX_TIME and end>0):
                start = 0
                #start = max(0, end - int(SALIENT_MOMENT_DURATION/dt));
                print "Play test %d, solver %d from %.2f s to %.2f s" % (i, s, start*dt, end*dt);
                #sleep(1.0);
                jp0 = data['q'][s][:,start]
                jp0[0:3] = ROBOT_MOVE + jp0[0:3]
                viewer.updateRobotConfig(jp0);
                if(DUMP_IMAGE_FILES):
                    viewer.startCapture('solver_'+str(s),'jpeg',pa+'/');
                jp = data['q'][s][:,start:end]
                jp[:,0:3] = ROBOT_MOVE + jp[:,0:3]


                viewer.playModified(jp, dt, cp_ctrl=cp_ctrl_i[s],cp=cp_i[s],vdcom=vdcom,solver= s,slow_down_factor=SALIENT_MOMENT_SLOWDOWN_FACTOR);    
                #viewer.play(data['q'][start:end,s,:], dt, SALIENT_MOMENT_SLOWDOWN_FACTOR);
                if(DUMP_IMAGE_FILES):
                   viewer.stopCapture();  
                   
            
    i += 1;
    if(i>=N_DIR):
        break;
        
time_to_fall = time_to_fall[:i,:];
#cmpare = np.hstack((itmax*dt,tmax,time_to_fall))    
nonfalls = abs(falls-np.ones(falls.shape))
#enf = task_error*nonfalls;
#real_falls = ((enf>0.20)*1+falls)
#real_falls = falls
real_nonfalls = abs(real_falls-np.ones(real_falls.shape))
error_at_nonfalls =task_error*real_nonfalls
time_taken_nonfalls = tmax*real_nonfalls
#error_at_zerovel = task_error*
avg_time =  time_to_reach.sum(axis=0)/(N_TESTS - real_falls.sum(axis=0))
avg_error = task_error.sum(axis=0)/(N_TESTS - real_falls.sum(axis=0))

if(SAVE_RESULTS):
    np.savez(plot_utils.FIGURE_PATH+'results', time_to_fall=time_to_fall);    

info = "Number of Tests run = "+str(runs)+'\n';
info += "====================================="+'\n';
info += "Number of fails detected "+'\n';
info += "====================================="+'\n';
info += "Classic Controller = "+str(falls.sum(axis=0)[0])+'\n';
info += "Robust Controller = "+str(falls.sum(axis=0)[1])+'\n';
info += "Robust Controller with Velocity Uncertainties= "+str(falls.sum(axis=0)[2])+'\n\n';
info += "Mean task error for both solvers:\n";
info += "====================================="+'\n';
info += "Classic Controller(in m) = "+str(avg_error[0])+'\n';
info += "Robust Controller(in m)  = "+str(avg_error[1])+'\n';
info += "Robust Controller with Uncertainties (in m)  = "+str(avg_error[2])+'\n\n';
info += "Mean Task time for both solvers:\n"
info += "====================================="+'\n';
info += "Classic Controller(in secs) = "+str(avg_time[0])+'\n';
info += "Robust Controller(in secs)  = "+str(avg_time[1])+'\n';
info += "Robust Controller with uncertainties(in secs)  = "+str(avg_time[2])+'\n\n';

#info += "Scores:\n"+str(np.sum(scores,0))+'\n';

print info_tail
print '\n'+info

if(SAVE_RESULTS):
    tfile = open(plot_utils.FIGURE_PATH+TEXT_FILE_NAME, "w")
    tfile.write(info);
    tfile.write('\n'+info_tail);
    tfile.close();

plt.show();
