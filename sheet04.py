import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
    @brief Models a sensor  
"""
class Sensor:
    
    """
        @brief Constructs a new sensor
        @param position The position of the sensor
        @param noise The measurement noise of the sensor
    """
    def __init__(self, position, noise):
        self.position = position
        self.noise = noise
        
    """
        @brief produces measurements using the groundtruth x
        @param x The groundtruth trajectory (2D projection)
        @return (2 x x.shape[1]) matrix with the measurements
    """
    def measure(self, x):
        dist = x - np.repeat(self.position[:2].reshape(2,1),x.shape[1],axis=1)
        
        n = np.random.normal(0,1,x.shape[1])
        
        r = np.sqrt(dist[0]**2 + dist[1]**2) + n*self.noise[0]
        phi = np.arctan2(dist[1],dist[0]) + n*self.noise[1] 
        
        result = np.concatenate((r.reshape(1,x.shape[1]),phi.reshape(1,x.shape[1])),axis=0)
        
        return result
    
"""
    @brief Converts polar coordinates to cartesian
    @param inp The input
    @param The result in cartesian coordinates
"""
def toCartesian(inp):
    x = inp[0]*np.cos(inp[1]).reshape(1,inp.shape[1])
    y = inp[0]*np.sin(inp[1]).reshape(1,inp.shape[1])
    
    cart = np.concatenate((x,y),axis=0)
    
    return cart

"""
    @brief Generates the trajectory
    @param params The parameters from the exercise sheet
    @param step_size How many values of the trajectory should be computed
    @return 3xstep_size matrix with the trajectory
"""
def trajectory(params, step_size=1000):
    v = params[0]
    ax = params[1]
    ay = params[2]
    az = params[3]
    
    T = ax/v
    t = np.linspace(0,T,step_size)
    
    x = v*t
    y = ay*np.sin((4.0*np.pi*v)/ax*t)
    z = az*np.sin((np.pi*v)/ax*t)
    
    return np.vstack((np.vstack((x,y)),z))

    
params = [20, 10, 1, 1]
traj = trajectory(params)

groundtruth = traj[0:2,:]

noise = np.diag([0.01, np.deg2rad(0.1)])

s1 = Sensor(np.array([0,100,10]),np.diag(noise))

measurement_polar = s1.measure(groundtruth)

measurement = toCartesian(measurement_polar)+np.repeat(s1.position[:2].reshape(2,1),groundtruth.shape[1],axis=1)

###################################################
#                DYNAMICS MODEL                   #
###################################################
#Piecewise constant white acceleration
delta_T = 0.5/1000 #Half an our divided by the number of measurements
F = np.eye(4)
F[:2,2:] = np.eye(2)*delta_T
D = np.zeros((4,4))
D[:2,:2]=0.25*np.eye(2)*delta_T**4
D[:2,2:]=0.5*np.eye(2)*delta_T**3
D[2:,:2]=0.5*np.eye(2)*delta_T**3
D[2:,2:]=np.eye(2)*delta_T
D*=10000    #Sigma^2 of the acceleration distribution
print(F)
print(D)

###################################################
#                INITIALIZATION                   #
###################################################
filter_results = []
predicted_results = []
filter_covariants = []
predicted_covariants = []

#Initialize with the first measurement
#Initial P has large variances
x = np.hstack((measurement[:,0],np.zeros(2)))
P = np.eye(4)*10
filter_results.append(x)
#Projection
H = np.zeros((2,4))
H[0,0] = 1
H[1,1] = 1
print(H)

###################################################
#                  KALMAN                         #
###################################################
for k in range(1,measurement.shape[1]):
    ###################################################
    #                 PREDICTION                      #
    ###################################################
    x = F@x
    P = F@P@F.T+D
    
    predicted_results.append(x)
    predicted_covariants.append(P)
    ###################################################
    #                 FILTERING                       #
    ###################################################
    z = measurement[:,k]
    
    #Convert radial error covariance to cartesian
    z_ = measurement_polar[:,k]
    phi = z_[1]
    D_phi = np.array([[np.cos(phi), -np.sin(phi)],
                      [np.sin(phi), np.cos(phi)]])
    r = z_[0]
    #Noise stored the standard deviations -> square to get variances
    R = (noise.copy())**2
    R[1,1] *= r**2
    R = D_phi@R@D_phi.T
    
    #Filter step
    v = z - H@x
    S = H@P@H.T + R
    W = P@H.T@np.linalg.inv(S)
    
    x = x + W@v
    P = P - W@S@W.T
    
    filter_results.append(x)
    filter_covariants.append(P)


###################################################
#                 Retrodiction                    #
###################################################
#Initialize retrodiction with last filter result
retrodicted_results = [filter_results[-1]]
retrodicted_covariants = [filter_covariants[-1]]

for i in range(0,measurement.shape[1]):
    k = len(filter_covariants)-i-1
    W = filter_covariants[k]@F.T@np.linalg.inv(predicted_covariants[k])
    
    x = filter_results[k]+W@(retrodicted_results[i]-predicted_results[k])
    P = filter_covariants[k]+W@(retrodicted_covariants[i]-predicted_covariants[k])@W.T
    
    retrodicted_results.append(x)
    retrodicted_covariants.append(P)


filter_results = np.array(filter_results)
retrodicted_results = np.array(retrodicted_results)


###################################################
#                 Plotting                        #
###################################################
#This is best enjoyed in full screen.

fig = plt.figure(figsize=(15,10),constrained_layout=True)

gs = fig.add_gridspec(2,2)

f_ax1 = fig.add_subplot(gs[:,0])
f_ax2 = fig.add_subplot(gs[0,1])
f_ax3 = fig.add_subplot(gs[1,1])

f_ax1.plot(measurement[0], measurement[1], label="Measurement s1",alpha=0.2,color="blue")
f_ax1.plot(filter_results[:,0], filter_results[:,1], label="Filtered", alpha=0.5,color="orange")
f_ax1.plot(retrodicted_results[:,0], retrodicted_results[:,1], label="Retrodiction", alpha=0.5,color="green")


f_ax1.plot(groundtruth[0], groundtruth[1], label="Groundtruth",alpha=0.5,color="red")

f_ax1.set_xlabel("x")
f_ax1.set_ylabel("y")

f_ax1.legend()

#####################################################################
f_ax2.plot(measurement[0], measurement[1], label="Measurement s1",alpha=0.2,color="blue")
f_ax2.plot(filter_results[:,0], filter_results[:,1], label="Filtered", alpha=1,color="orange")

for i,P in enumerate(filter_covariants):
    #Get size and orientation of the covariance ellipse using evd
    eigw,eigv = np.linalg.eig(P[:2,:2])
    
    angle = np.arctan2(eigv[1,0],eigv[0,0])+np.pi
    angle_deg = np.rad2deg(angle)
    
    el = Ellipse((filter_results[i,0],filter_results[i,1]), 2*np.sqrt(eigw[0]),2*np.sqrt(eigw[1]),angle_deg)
    
    f_ax2.add_artist(el)
    el.set_facecolor(np.array([1,0.8,0]))

f_ax2.plot(groundtruth[0], groundtruth[1], label="Groundtruth",alpha=0.5,color="red")

f_ax2.set_xlabel("x")
f_ax2.set_ylabel("y")

f_ax2.legend()

#####################################################################
f_ax3.plot(measurement[0], measurement[1], label="Measurement s1",alpha=0.2,color="blue")
f_ax3.plot(retrodicted_results[:,0], retrodicted_results[:,1], label="Retrodiction", alpha=1,color="green")

for i,P in enumerate(retrodicted_covariants):
    #Get size and orientation of the covariance ellipse using evd
    eigw,eigv = np.linalg.eig(P[:2,:2])
    
    angle = np.arctan2(eigv[1,0],eigv[0,0])+np.pi
    angle_deg = np.rad2deg(angle)
    
    el = Ellipse((retrodicted_results[i,0],retrodicted_results[i,1]), 2*np.sqrt(eigw[0]),2*np.sqrt(eigw[1]),angle_deg)
    
    f_ax3.add_artist(el)
    el.set_facecolor(np.array([0,1,0]))

f_ax3.plot(groundtruth[0], groundtruth[1], label="Groundtruth",alpha=0.5,color="red")

f_ax3.set_xlabel("x")
f_ax3.set_ylabel("y")

f_ax3.legend()

plt.show()