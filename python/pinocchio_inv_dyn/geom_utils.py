#from polytope_conversion_utils import *
from numpy import zeros, sqrt, array, vstack
import numpy as np
#from math import cos, sin, tan, atan, pi
import matplotlib.pyplot as plt
import cdd
import plot_utils as plut
from polytope_conversion_utils import poly_face_to_span,poly_span_to_face,NotPolyFace,arbitrary_face_to_span

NUMBER_TYPE = 'float'  # 'float' or 'fraction'

''' Compute the projection matrix of the cross product.
'''
def crossMatrix( v ):
    VP = np.array( [[  0,  -v[2], v[1] ],
                    [ v[2],  0,  -v[0] ],
                    [-v[1], v[0],  0   ]] );
    return VP;
    
''' Check whether v is inside a 3d cone with the specified normal direction
    and friction coefficient. 
'''
def is_vector_inside_cone(v, mu, n):
    P = np.eye(3) - np.outer(n, n);
    return (np.linalg.norm(np.dot(P,v)) - mu*np.dot(n,v)<=0.0);

    
''' Find the intersection between two lines:
        a1^T x = b1
        a2^T x = b2
'''
def find_intersection(a1, b1, a2, b2):
    x = np.zeros(2);
    den = (a1[0]*a2[1] - a2[0]*a1[1]);
    if(abs(den)<1e-6):
        print "ERROR: Impossible to find intersection between two lines that are parallel";
        return x;
        
    if(np.abs(a1[0])>np.abs(a2[0])):
        x[1] = (-a2[0]*b1 + a1[0]*b2)/den;
        x[0] = (b1-a1[1]*x[1])/a1[0];
    else:
        x[1] = (-a2[0]*b1 + a1[0]*b2)/den;
        x[0] = (b2-a2[1]*x[1])/a2[0];        
    return x;
    
''' Find the line passing through two points:
        a^T x1 + b = 0
        a^T x2 + b = 0
'''
def find_line(x1, x2):
    den = (x1[0]*x2[1] - x2[0]*x1[1]);
    if(abs(den)<1e-4):
#        print "ERROR: x1 and x2 are too close, x1=(%f,%f), x2=(%f,%f)" % (x1[0],x1[1],x2[0],x2[1]);
        return (zeros(2),-1);
#    a = np.array([-(x1[1] - x2[1])/den, -(x2[0] - x1[0])/den]);
#    a_norm = np.linalg.norm(a);
#    a /= a_norm;
#    b = -1.0/a_norm;

    a = np.array([x2[1]-x1[1], x1[0]-x2[0]]);
    a /= np.linalg.norm(a);
    b = -a[0]*x1[0] - a[1]*x1[1];
#    print "a=(%f,%f), a2=(%f,%f), b=%f, b2=%f" % (a[0],a[1],a2[0],a2[1],b,b2);
    return (a,b);

    
''' Compute the area of a 2d triangle with vertices a, b and c. 
'''
def compute_triangle_area(a, b, c):
    la = np.linalg.norm(a-b);
    lb = np.linalg.norm(b-c);
    lc = np.linalg.norm(c-a);
    s = 0.5*(la+lb+lc);
    return sqrt(s*(s-la)*(s-lb)*(s-lc));

    
''' Plot inequalities F_com*x+f_com=0 on x-y plane.
'''
def plot_inequalities(F_com, f_com, x_bounds, y_bounds, ls='--', color='k', ax=None, lw=8):
#    if(F_com.shape[1]!=2):
#        print "[ERROR in plot_inequalities] matrix does not have 2 columns";
#        return;

#    if(F_com.shape[0]!=len(f_com)):
#        print "[ERROR in plot_inequalities] matrix and vector does not have the same number of rows";
#        return;

    if(ax==None):
        f, ax = plut.create_empty_figure();
    com = np.zeros(2);     # com height
    com_x = np.zeros(2);
    com_y = np.zeros(2);
    for i in range(F_com.shape[0]):
        if(np.abs(F_com[i,1])>1e-13):
            com_x[0] = x_bounds[0];   # com x coordinate
            com_x[1] = x_bounds[1];   # com x coordinate
            com[0] = com_x[0];
            com[1] = 0;
            com_y[0] = (-f_com[i] - np.dot(F_com[i,:],com) )/F_com[i,1];
            
            com[0] = com_x[1];
            com_y[1] = (-f_com[i] - np.dot(F_com[i,:],com) )/F_com[i,1];
            #plt.ylim( y_bounds[0]-1, y_bounds[1]+1 )
            ax.plot(com_x, com_y, ls=ls, color=color, linewidth=lw);
        elif(np.abs(F_com[i,0])>1e-13):
            com_y[0] = y_bounds[0];
            com_y[1] = y_bounds[1];
            com[0] = 0;
            com[1] = com_y[0];
            com_x[0] = (-f_com[i] - np.dot(F_com[i,:],com) )/F_com[i,0];
    
            com[1] = com_y[1];
            com_x[1] = (-f_com[i] - np.dot(F_com[i,:],com) )/F_com[i,0];
            ax.plot(com_x, com_y, ls=ls, color=color, linewidth=lw);
        else:
            pass;
#            print "[WARNING] Could not print one inequality as all coefficients are 0: F_com[%d,:]=[%f,%f]" % (i,F_com[i,0],F_com[i,1]);
    return ax;

''' Plot the polytope A*x+b>=0 with vectices V '''
def plot_polytope(A, b, V=None, color='b', ax=None, plotLines=False, lw=2,label=None,dots=True):
    
    # find points   
    (points, I) = arbitrary_face_to_span(-A, b);  
     
    if(ax==None):
        f, ax = plut.create_empty_figure();
        
    n = b.shape[0];     
    if(n<2):
        return (ax,None);
    if(V==None):
        V = np.zeros(((points.shape[1])*2,2));
        '''
        for i in range(n):
            V[i,:] = find_intersection(A[i,:], b[i], A[(i+1)%n,:], b[(i+1)%n]);
        '''
        index = 0
        # Find line segments
        for i in range(n): 
            line = check_point_in_line(A[i,:], b[i],points)
            line = line[0:2,:]            
            if  str(type(line)) == "<type 'numpy.ndarray'>":
                V[index:index+2,:] = line;
                index+=2;
      
                 
#    print "Polytope vertices:", V
    min_x = min(points[0,:]) ; min_y = min(points[1,:]) ;
    
    max_x = max(points[0,:]) ; max_y = max(points[1,:]) ;
    #print '-----'
    #print min_x,max_x;
    #print min_y,max_y;
    if(plotLines):
        plot_inequalities(A, b, [min_x,max_x], [min_y,max_y], color=color, ls='--', ax=ax, lw=lw);
    n = b.shape[0];    
    if(n<2):
        return (ax,None);
        
    xx = np.zeros(2);
    yy = np.zeros(2); 
    c = color;
    for i in range(0,V.shape[0],2):
        xx[0] = V[i,0];
        xx[1] = V[(i+1),0];
        yy[0] = V[i,1];
        yy[1] = V[(i+1),1];
        if(c == 'ordered'):
            color = COLOR[(i/2)]
        if (label == None) or (i<(V.shape[0]-2)):
            line, = ax.plot(xx, yy, color=color, ls='-', lw=2*lw);
        else:
            line, = ax.plot(xx, yy, color=color, ls='-', lw=2*lw, label=label);
    
    if(dots): 
        ax.scatter(points[0,:],points[1,:],s=100,color=color)
    return (ax, line);

    
def compute_convex_hull(S):
    """
    Returns the matrix A and the vector b such that:
        {x = S z, sum z = 1, z>=0} if and only if {A x + b >= 0}.
    """
    V = np.hstack([np.ones((S.shape[1], 1)), S.T])
    # V-representation: first column is 0 for rays, 1 for vertices
    V_cdd = cdd.Matrix(V, number_type=NUMBER_TYPE)
    V_cdd.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(V_cdd)
    H = np.array(P.get_inequalities())
    b, A = H[:, 0], H[:, 1:]
    return (A,b)

def auto_generate_polytopes(num_points,interval,extras=None):
    '''
    Auto generate 2 dimensional inequalities from randomly generated points
    '''
    dim = len(interval)
    xinterval = interval[0]
    yinterval = interval[1]
    zinterval = interval[2]
    x = np.random.uniform(xinterval[0],xinterval[1],(1,num_points));
    y = np.random.uniform(yinterval[0],yinterval[1],(1,num_points));
    print xinterval   
    xint = [min(x[0]),max(x[0])]
    yint = [min(y[0]),max(y[0])]
    if dim > 2:
        zinterval = interval[2]
        z = np.random.uniform(zinterval[0],zinterval[1],(1,num_points));
        V = np.vstack((x,y,z));
        zint = [min(z[0]),max(z[0])]
        interval = [xint,yint,zint]
    else:
        V = np.vstack((x,y));
        interval = [xint,yint]
    #print 'V shape before conversion'  
    if extras != None:
        V = np.hstack((V,extras.T))
    #print V.shape
    A,b= poly_span_to_face(V);
    V= poly_face_to_span(A,b);
    #print 'V shape after conversion' 
    #print V.shape
    A,b= poly_span_to_face(V);
    return A,b,V,interval

def check_point_polytope(A,b,p):
    d = np.dot(A,p).T+b;
    s = np.ones(b.shape[0])*-1e-3
    res = np.prod(d >= s);
    if res == 1:
        return d,True
    else:
        return d,False

def twodprojection(V):
    xyproj = V[0:2,:]  
    A,b= poly_span_to_face(xyproj);
    V= poly_face_to_span(A,b);
    return -A,b

def check_point_in_line(A,b,p):
    ''' 
    Find two points from the input set 'p' that solves A*p + b = 0
    
    '''
    j = 0;
    tmp = np.zeros((1,A.shape[0]));
    for i in range(p.shape[1]):
        if A.shape[0] >2:
            point = np.array([p[0][i],p[1][i],p[2][i]])
        else:
            point = np.array([p[0][i],p[1][i]])
        #print point
        sol = np.dot(A,point)+b;            
        if sol < 1e-6:
            tmp = np.resize(tmp,((j+1),A.shape[0]));    
            tmp[j,:] = point;
            j = j + 1       
    return tmp    
   