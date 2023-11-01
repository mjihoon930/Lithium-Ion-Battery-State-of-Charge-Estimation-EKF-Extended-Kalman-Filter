#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:03:57 2023

@author: moonjihoon
"""

import numpy as np
import math
import sympy
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

## Input Current Profile
amplitude = 0.5
start_times = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000]
durations = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
end_time = 3000
num_samples = 3000

time = np.linspace(0, end_time, num_samples)

U = np.zeros_like(time)

for start, duration in zip(start_times, durations):
    U[(time >= start) & (time < start + duration)] = amplitude
    
## Initial Values
N = 3000                                # Number of time steps
dt = 1

# Initial SOC
SOC_int = 0.6;

x0 = np.array([[SOC_int],[0],[0]])      # Initial State
xhat = x0                               # Initial State Estimate
Q = np.diag(np.array([0.000000005, 0.000000005, 0.000000005]))        # Process Noise Covariance
R = np.diag(np.array([0.000000005]))              # Measurement Noise Covariance
P0 = np.diag(np.array([1, 1, 1]))       # Initial Error Covariance

## Linearization

x1 = sympy.Symbol('x1')
x2 = sympy.Symbol('x2')
x3 = sympy.Symbol('x3')
u = sympy.Symbol('u')

# Nonlinear Parameters
V_oc = -1.031*sympy.exp(-35*x1) + 3.685 + 0.2156*x1 - 0.1178*x1**2 + 0.3201*x1**3
R0 = 0.1562*sympy.exp(-24.37*x1) + 0.07446
Rs = 0.3208*sympy.exp(-29.14*x1) + 0.04669
Cs = -752.9*sympy.exp(-13.51*x1) + 703.6
Rf = 6.603*sympy.exp(-155.2*x1) + 0.04984
Cf = -6056*sympy.exp(-27.12*x1) + 4475
Rsd = 999999999999
Cb = 3060

## Nonlinear Model
# State Equations
f1 = -(1/(Rsd*Cb))*x1 - (1/Cb)*u
f2 = -(1/(Rf*Cf))*x2 + (1/Cf)*u
f3 = -(1/(Rs*Cs))*x3 + (1/Cs)*u

# Output Equation
Vb = V_oc - R0*u - x2 - x3;

## Linearized Model Matrix
# F and G matrix
df1dx1 = sympy.diff(f1,x1)
df2dx1 = sympy.diff(f2,x1)
df2dx2 = sympy.diff(f2,x2)
df3dx1 = sympy.diff(f3,x1)
df3dx3 = sympy.diff(f3,x3)

df1du = sympy.diff(f1,u)
df2du = sympy.diff(f2,u)
df3du = sympy.diff(f3,u)

F11 = df1dx1.subs(x1,SOC_int)
F21 = df2dx1.subs({x1:SOC_int , x2:0 , u:0})
F22 = df2dx2.subs({x1:SOC_int , x2:0})
F31 = df3dx1.subs({x1:SOC_int , x3:0 , u:0})
F33 = df3dx3.subs({x1:SOC_int , x3:0})

G1 = df1du
G2 = df2du.subs(x1,SOC_int)
G3 = df3du.subs(x1,SOC_int)

# H and D matrix
dVbdx1 = sympy.diff(Vb,x1)
dVbdx2 = sympy.diff(Vb,x2)
dVbdx3 = sympy.diff(Vb,x3)

dVbdu = sympy.diff(Vb,u)

H1 = dVbdx1.subs({x1:SOC_int, u:0})
H2 = dVbdx2
H3 = dVbdx3

# Define System Matrices
F = np.array([[F11, 0, 0],[F21, F22, 0],[F31, 0, F33]])
G = np.array([[G1],[G2],[G3]])
H = np.array([H1, H2, H3])
D = dVbdu.subs(x1,SOC_int)

H0 = H
h0 = Vb.subs({x1:SOC_int, x2:0, x3:0, u:0})
z0 = h0 - H0 @ x0

# Discrete Time Extended Kalman Filter
class DiscreteTimeKF:
    
    def __init__(self, State, U, State_covariance, F, H, Q, R, G, M, h, L,Rs,Cs,Rf,Cf,Rsd,Cb,dt):
        self.State = State
        self.Measurement = None
        self.Control = U
        self.StateCovariance = State_covariance
        self.ProcessCovariance = Q
        self.MeasurementCovariance = R
        self.Fmatrix = F
        self.Gmatrix = G
        self.Hmatrix = H
        self.Mmatrix = M
        self.hmatrix = h
        self.Lmatrix = L
        self.R_s = Rs
        self.C_s = Cs
        self.R_f = Rf
        self.C_f = Cf
        self.R_sd = Rsd
        self.C_b = Cb
        self.d_t = dt
    
    def predict(self):
        """
        PREDICT the state estimate and compute its covariance
        """
        xhat = self.State
        P = self.StateCovariance
        F = self.Fmatrix
        Q = self.ProcessCovariance
        U = self.Control
        L = self.Lmatrix
        Rs = self.R_s
        Cs = self.C_s
        Rf = self.R_f
        Cf = self.C_f
        Rsd = self.R_sd
        Cb = self.C_b
        d_t = self.d_t
        
        xhat[0] = xhat[0] + d_t*(-(1/(Rsd*Cb))*xhat[0] - (1/Cb)*U);
        xhat[1] = xhat[1] + d_t*(-(1/(Rf*Cf))*xhat[1] + (1/Cf)*U);
        xhat[2] = xhat[2] + d_t*(-(1/(Rs*Cs))*xhat[2] + (1/Cs)*U);
            
        P = F.dot(P).dot(F.T) + L @ Q @ L.T
        
        self.StateCovariance = P
        self.State = xhat
    
    def correct(self):
        """
        CORRECT the state estimate and compute its covariance
        """
        xhat = self.State
        P = self.StateCovariance
        y = self.Measurement
        H = self.Hmatrix
        h = self.hmatrix
        M = self.Mmatrix
        R = self.MeasurementCovariance
        
        K = P @ H.T / (H @ P @ H.T + M * R * M)
        xhat = xhat + K.T*(y - h)
        P = (np.eye(len(xhat)) - K.dot(H)) * P
        
        self.StateCovariance = P
        self.State = xhat

# Create an object of the class using the class name
#from RUN_KF import DiscreteTimeKF

KF = DiscreteTimeKF

KF.State = xhat
KF.Control = U[1]
KF.StateCovariance = P0
KF.ProcessCovariance = Q
KF.MeasurementCovariance = R
KF.Fmatrix = F
KF.Gmatrix = G
KF.Hmatrix = H0
KF.Mmatrix = 1
KF.hmatrix = h0
KF.Lmatrix = np.eye(3)
KF.R_s = Rs.subs(x1,SOC_int)
KF.C_s = Cs.subs(x1,SOC_int)
KF.R_f = Rf.subs(x1,SOC_int)
KF.C_f = Cf.subs(x1,SOC_int)
KF.R_sd = Rsd
KF.C_b = Cb
KF.d_t = dt

## Simulate system with noisy measurements and call Kalman filter class
# Preallocate for storage and store initial conditions

xarray = np.zeros((3,N))
xarray[:,0] = x0.flatten()
yarray = np.zeros((1,N))
xhatarray = np.zeros((3,N))
xhatarray[:,0] = x0.flatten()
yhatarray = np.zeros((1,N))

Parray = np.zeros((3,N))
Parray[:,0] = np.zeros((3,1)).flatten()

x = np.zeros((3,N))
y = np.zeros((1,N))

x[:,0] = x0.flatten()
y[:,0] = -1.031*math.exp(-35*x[0,0])+3.685+0.2156*x[0,0] - 0.1178*x[0,0]**2 + 0.3201*x[0,0]**3 - (0.1562*math.exp(-24.37*x[0,0])+0.07446)*U[0] - x[1,0] - x[2,0];

x_true = np.zeros((3,N))
y_true = np.zeros((1,N))

x_true[:,0] = x0.flatten()
y_true[:,0] = y[:,0]

P = P0
H = H0
h = h0
z = z0

yarray[:,0] = y[:,0]
yhatarray[:,0] = y[:,0]


for i in range(1,N):
    
    v = sqrtm(R) @ np.random.randn(1)
    w = sqrtm(Q) @ np.random.randn(3,1)
    
    
    R_s = 0.3208*math.exp(-29.14*x[0,i-1])+0.04669;
    C_s = -752.9*math.exp(-13.51*x[0,i-1])+703.6;
    R_f = 6.603*math.exp(-155.2*x[0,i-1])+0.04984;
    C_f = -6056*math.exp(-27.12*x[0,i-1])+4475;
    
    x[0,i] = x[0,i-1] + dt*(-(1/(Rsd*Cb))*x[0,i-1] - (1/Cb)*U[i-1]) + w[0];
    x[1,i] = x[1,i-1] + dt*(-(1/(R_f*C_f))*x[1,i-1] + (1/C_f)*U[i-1]) + w[1];
    x[2,i] = x[2,i-1] + dt*(-(1/(R_s*C_s))*x[2,i-1] + (1/C_s)*U[i-1]) + w[2];
    
    y[:,i] =  -1.031*math.exp(-35*x[0,i])+3.685 + 0.2156*x[0,i] - 0.1178*x[0,i]**2 + 0.3201*x[0,i]**3 - (0.1562*math.exp(-24.37*x[0,i])+0.07446)*U[i] - x[1,i] - x[2,i] + v;
    
    x_true[0,i] = x_true[0,i-1] + dt*(-(1/(Rsd*Cb))*x_true[0,i-1] - (1/Cb)*U[i-1]) 
    x_true[1,i] = x_true[1,i-1] + dt*(-(1/(R_f*C_f))*x_true[1,i-1] + (1/C_f)*U[i-1])
    x_true[2,i] = x_true[2,i-1] + dt*(-(1/(R_s*C_s))*x_true[2,i-1] + (1/C_s)*U[i-1])
    
    y_true[:,i] =  -1.031*math.exp(-35*x_true[0,i])+3.685 + 0.2156*x_true[0,i] - 0.1178*x_true[0,i]**2 + 0.3201*x_true[0,i]**3 - (0.1562*math.exp(-24.37*x_true[0,i])+0.07446)*U[i] - x_true[1,i] - x_true[2,i] 
    
    xarray[:,i] = x[:,i].flatten()
    yarray[:,i] = y[:,i].flatten()
    
    KF.Measurement = y[:,i]
    KF.Control = U[i]
    
    KF.R_s = R_s
    KF.C_s = C_s
    KF.R_f = R_f
    KF.C_f = C_f
    
    #Predict
    KF.predict(KF)
    
    x_1 = float(KF.State[0])
    x_2 = float(KF.State[1])
    x_3 = float(KF.State[2])
    
    dVbdx1 = (7217*math.exp(-35*x_1))/200 - (589*x_1)/2500 + (1903297*U[i]*math.exp(-(2437*x_1)/100))/500000 + (9603*x_1**2)/10000 + 539/2500;
    dVbdx2 = -1;
    dVbdx3 = -1;
    
    H = np.array([dVbdx1, dVbdx2, dVbdx3])
    h = -1.031*math.exp(-35*x_1)+3.685 + 0.2156*x_1-0.1178*x_1**2 + 0.3201*x_1**3 - (0.1562*math.exp(-24.37*x_1)+0.07446)*U[i] - x_2 - x_3
    
    KF.Hmatrix = H
    KF.hmatrix = h
    
    # Update
    KF.correct(KF)
    
    # Save estimates
    xhatarray[:,i] = KF.State.flatten()
    yhatarray[:,i] = -1.031*math.exp(-35*xhatarray[0,i])+3.685 + 0.2156*xhatarray[0,i]-0.1178*xhatarray[0,i]**2 + 0.3201*xhatarray[0,i]**3 - (0.1562*math.exp(-24.37*xhatarray[0,i])+0.07446)*U[i] - xhatarray[1,i] - xhatarray[2,i]
    Parray[:,i] = np.diag(KF.StateCovariance)

e = xarray - xhatarray

# Figure
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

t = np.arange(N)

#axs[0].plot(t, x_true[0], 'r', linewidth=3)
axs[0].plot(t, xarray[0], 'r', linewidth=3)
axs[0].plot(t, xhatarray[0], 'b', linewidth=3, linestyle='dotted')
axs[0].set_ylabel('x_1', fontsize=14)
axs[0].legend(['True Value', 'Estimated'], fontsize=14)
axs[0].set_title('Kalman Filter Estimation for x_1', fontsize=14)

#axs[1].plot(t, x_true[1], 'r', linewidth=3)
axs[1].plot(t, xarray[1], 'r', linewidth=3)
axs[1].plot(t, xhatarray[1], 'b', linewidth=3, linestyle='dotted')
axs[1].set_ylabel('x_2', fontsize=14)
axs[1].legend(['True Value', 'Estimated'], fontsize=14)
axs[1].set_title('Kalman Filter Estimation for x_2', fontsize=14)

#axs[2].plot(t, x_true[2], 'r', linewidth=3)
axs[2].plot(t, xarray[2], 'r', linewidth=3)
axs[2].plot(t, xhatarray[2], 'b', linewidth=3, linestyle='dotted')
axs[2].set_ylabel('x_3', fontsize=14)
axs[2].legend(['True Value', 'Estimated'], fontsize=14)
axs[2].set_title('Kalman Filter Estimation for x_3', fontsize=14)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(8, 8))
#axs.plot(t, y_true.T, 'r', linewidth=3)
axs.plot(t, yarray.T, 'r', linewidth=3)
axs.plot(t, yhatarray.T, 'b', linewidth=3, linestyle='dotted')
axs.set_ylabel('y', fontsize=14)
axs.legend(['True Value', 'Estimated'], fontsize=14)
axs.set_title('Kalman Filter Estimation for y', fontsize=14)
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
axs[0].plot(t, e[0], 'b', linewidth=3)
axs[0].set_ylabel('e_1', fontsize=14)
axs[0].legend(['error 1'], fontsize=14)
axs[0].set_title('Kalman Filter Estimation Error for x_1', fontsize=14)


axs[1].plot(t, e[1], 'b', linewidth=3)
axs[1].set_ylabel('e_2', fontsize=14)
axs[1].legend(['error 2'], fontsize=14)
axs[1].set_title('Kalman Filter Estimation Error for x_2', fontsize=14)

axs[2].plot(t, e[2], 'b', linewidth=3)
axs[2].set_ylabel('e_3', fontsize=14)
axs[2].legend(['error 3'], fontsize=14)
axs[2].set_title('Kalman Filter Estimation Error for x_3', fontsize=14)

plt.tight_layout()
plt.show()



