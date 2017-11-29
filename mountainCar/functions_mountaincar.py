# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:23:52 2017

@author: Ryutaro
"""
import numpy as np
import gym
from gym import spaces
from scipy import linalg as LA #for inverse matrix
from gym import spaces

#hoge = [0, 0.25, 0.5, 1]
hoge = [0.125, 0.375, 0.625, 0.875]
xBar = np.zeros([16,2])
for i in range(4):
    for j in range(4):
        xBar[4*i+j,:] = [hoge[i], hoge[j]]

invSigGrid = np.matrix([[9.4675, 0],[0, 9.4675]])#check MATLAB mountain_car_calc_score.m

###############################################################################
def evaluate_policy(theta,actionSet, env, nActions):
    total_reward = 0    
    s = env.reset()
    s = normalizeS(s, env)
    while(True):        
        #predict array of action probabilities
        probs = policy(s, theta, xBar, nActions)# probability of actions given states    
        a = np.random.choice(actionSet,p = probs)
        new_s,r,done,info = env.step(a)
        s = normalizeS(new_s, env)
        total_reward += r
        #s = new_s
        if done:
            #states.append(s)
            #if t == (t_max-1):
                #print("NOT reach the goal")
            #else:
                #print("reach the goal")
            return total_reward
    #print("NOT reach the goal")
    return total_reward
    
###############################################################################
def normalizeS(s,env):
    return (s -env.observation_space.low )/(env.observation_space.high - env.observation_space.low)
###############################################################################
def policy(s, theta, xBar, nActions):
    kappa =  1.3*0.25
    phi = np.diag(np.exp(- (s-xBar).dot(invSigGrid).dot((s-xBar).T) / (2*kappa**2))) 
    theta2 = theta.reshape([nActions,16])
    phi2 = theta2.dot(phi)
    prob = phi2/sum(phi2)
    return prob
    
###############################################################################     
def scoreFunc(s, a, theta):
    kappa = 1.3*0.25
    phi = np.diag(np.exp(- (s-xBar).dot((s-xBar).T) / (2*kappa*2))) 
    phi2 = np.array([])
    for i in actionset:
        if a==i:
            phi2 = np.append(phi2, phi)
        else:
            phi2 = np.append(phi2, np.zeros([16]))
    #phiTheta = theta.dot(phi2)
    theta2 = theta.reshape([n_actions,16])
    sumDenom = 0
    sumNum = np.zeros(16*n_actions)
    for i in range(n_actions):
        sumDenom += np.exp(phi.T.dot(theta2[i]))
        sumNum[16*i:16*i+16] += phi*np.exp(phi.T.dot(theta2[i]))     
    score = phi2 - sumNum/sumDenom
    return score