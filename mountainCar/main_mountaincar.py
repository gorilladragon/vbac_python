# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:55:11 2017
@author: Ryutaro
"""
###############################################################################
import numpy as np
import gym
from gym import spaces
from scipy import linalg as LA #for inverse matrix
from gym import spaces
import os
path = "C:/Users/Ryutaro/Desktop/vbac_python/mountainCar"
os.chdir(path)
from functions_mountaincar import *
###############################################################################
#check the environment
space = spaces.Discrete(8) # Set with 8 elements {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8


env = gym.make("MountainCar-v0")
print(env.action_space) #right or left
print(env.observation_space)# dim of observation
print(env.observation_space.high) #observation upper bounds
print(env.observation_space.low)  #observation lower bounds

observation = env.reset()
#Hyper parameters##############################################################
nStates = env.observation_space# number of states
nActions = env.action_space.n - 1# number of available actions
actionSet = [0,2] #push left/right respectively
learningRate = 1e-2
gamma = 0.99# reward discount factor
nTheta = 16*nActions

nTrial = 10
nUpd = 50
nEpi = 1
nEval = 100 #how many episodes to evaluate the policy

nAL = 10 #how often to do active learning

#main part##############################################################
for i in range(nTrial):
    #initialize policy parameters
    theta = np.random.uniform(0,1,nTheta)
    for j in range(nUpd):
        ###evaluate the current policy parameter
        evaluate = np.array([])
        for e in range(nEval):
            evaluate = np.append(evaluate, evaluate_policy(theta, actionSet,env))
        aveTR = np.mean(evaluate)#average total reward for current theta
        print("Trial: "+str(i)+" Update: "+str(j)+" Performance: "+str(aveTR))
        
        #initialize the data
        epiS = np.array([])
        epiA = np.array([])
        epiR = np.array([])
        epiScr = np.array([])
        


    return score
    
def Fisher(states, actions, theta):
    T = len(states)
    dtheta = len(theta)
    Ut = np.zeros([dtheta,T])
    for t in range(T):
        Ut[:,t] = scoreFunc(states[t],actions[t],theta)
    Gt = Ut.dot(Ut.T)/(T+1)
    return Ut, Gt

#kernels
def stateKernel(s1, s2, sigmak):
    return np.exp( -(s1-s2).T.dot(s1-s2)/(2*sigmak**2) )
def FisherKernel(s1, s2, a1, a2, theta):
    u1 = scoreFunc(s1, a1, theta)
    u2 = scoreFunc(s2, a2, theta)
    G = Fisher([s1, s2], [a1, a2], theta)[1]############################check if it's right
    Ginv = LA.inv(G + 10**(-5)*np.identity(n_actions*16))
    return u1.T.dot(G).dot(u2)
def kernel(s1, s2, a1, a2, theta, sigmak = 1.3*0.25):
    return stateKernel(s1, s2,sigmak) + FisherKernel(s1, s2, a1, a2, theta)
 
    
    
    
    
#########################Experiments#########################
###Experiments set up
M = 1
conv = 10**(-5)
theta = np.zeros([n_actions*16])
theta = np.random.normal(0,1, n_actions*16)
#this is for policy discretization
xBar = np.zeros([16,2])
for i in range(4):
   xBar[4*i:(4*i+4),:] =[[0.125+0.25*j, 0.125+0.25*i] for j in range(4)]
sigma2 = 0.1

BAC(theta,M,conv)
