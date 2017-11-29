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
#path = "C:/Users/Ryutaro/Desktop/vbac_python/mountainCar"
path = "C:/Users/workshop/Desktop/vbac_python/mountainCar"
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
nTheta = 16 * nActions
T = 200
alpha = 0.2 # learning rate

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
        ### evaluate the current policy parameter
        evaluate = np.array([])
        for e in range(nEval):
            evaluate = np.append(evaluate, evaluate_policy(theta, actionSet,env, nActions))
        aveTR = np.mean(evaluate)#average total reward for current theta
        print("Trial: "+str(i)+" Update: "+str(j)+" Performance: "+str(aveTR))
        
        ### initialize the data
        epiS = np.array([])
        epiA = np.array([])
        epiR = np.array([])
        epiScr = np.array([]) # score
        epiNS = np.array([]) #normalized staes
        tempNS = np.array([])
        tempA = np.array([])
        tempR = np.array([])
        tempScr = np.array([])
        
        countF = 0 # for computing fisher kernel
        
        ### inducing variable
        Nind = 30
        aInd = np.random.choice(actionSet, Nind) #inducing input for actions
        s1Ind = np.arange(0.0, 1.0, 1/Nind)
        np.random.shuffle(s1Ind) #inducing input for s1
        s2Ind = np.arange(0.0, 1.0, 1/Nind)
        np.random.shuffle(s2Ind) #inducing input for s2
        Bu = np.concatenate(([s1Ind], [s2Ind], [aInd]), axis = 0).T
        ### initialization for VBSGP
        pUmean = np.zeros(Nind)
        jitter = 10**(-5)
        # have to define pUCovariance  KsU + KfU
        
        qUmean = np.zeros(Nind)
        qUcovariance = np.identity(Nind)
        
        qHypermean = np.ones(2)
        qHypercovariance = np.identity(2)
        
        sigmaf = 0.5
        
        #prev
        qUpmean = qUmean
        qUpcovariance = qUcovariance
        qHyperpmean = qHypermean
        qHyperpcovariance = qHypercovariance
        sigmafp = sigmaf
        
        G = np.zeros([nTheta, nTheta])
        #repeat episodes nEpi times
        for k in range(nEpi):
            s = env.reset()
            s = normalizeS(s,env)
            news = s
            done = False
            for t in range(T):
                s = news
                ### not active learning (i.e. follow the policy)
                if(t%nAL != 0):
                    a = policy(s, theta, xBar, nActions)
                    scr = scoreFunc(s, a, theta)
                    G = G + scr.dot(scr.T)
                    countF += 1
                    epiScr = np.array(epiScr, scr)
                ### active learning
                else:
                    tempSize = tempS.shape[1]
                    Ht = np.zeros([tempSize,tempSize])
                    for tt in range(tempSize): 
                        Ht[tt,tt] = 1
                        if tt != (tempSize-1): 
                            Ht[tt,tt+1] = -gamma
                    tempG = G/countF + np.identity(nTheta)*10**(-5)
                    # cholesky decompostion to compute invG
                    
                    # temporary input
                    tempData
                    
                    # Update VBSGP parameters
                    #qUpmean, qUpcovariance, qHyperpmean, qHyperpcovariance, sigmafp = VBSGP()
                    # store old parameters
                    qUpmean = pUmean
                    qUpcovariance = pUcovariance
                    qHyperpmean = qHypermean
                    qHyperpcovariance = qHypercovariance
                    sigmafp = sigmaf
                    
                    # choose action based on AL
                    MI = np.zeros(nActions)
                    B0 = tempG
                    for count_a in range(nActions):
                        aa = actionSet[count_a]
                        #S = vsgp_fixed()
                        gradCov = B0 - (Bu.dot(invKuu).dot(Bu.T)) + (Bu.dot(invKuu)).dot(S).(Bu.dot(invKuu)).T
                        MI[count_a] = -np.linalg.det(gradCov) #determinant
                    a = np.argmax(MI)
                    tempNS = np.array([])
                    tempA = np.array([])
                    tempR = np.array([])
                    tempScr = np.array([])
                    tempData = np.array([])
                    news, r, done, info = env.step(a)
                    news = normalizeS(news,env)
                    
                    #append
                    epiS = np.append(epiS, s)
                    epiA = np.append(epiA, a)
                    epiR = np.append(epiR, r)
                    #epiNS = np.append(epiNS, s) #normalized staes
                    tempS = np.append(tempS, s)
                    tempA = np.append(tempA, a)
                    tempR = np.append(tempR, r)
                    if done:
                        break
        # update the parameters
        tempSize = tempS.shape[1]
        if(tempSize != 0):
            Ht = np.zeros([tempSize,tempSize])
            for tt in range(tempSize): 
                Ht[tt,tt] = 1
                if tt != (tempSize-1): 
                    Ht[tt,tt+1] = -gamma
            tempG = G/countF + np.identity(nTheta)*10**(-5)
            tempData
                    
            # Update VBSGP parameters
            #qUpmean, qUpcovariance, qHyperpmean, qHyperpcovariance, sigmafp = VBSGP()
            # store old parameters
            qUpmean = qUmean
            qUpcovariance = qUcovariance
            qHyperpmean = qHypermean
            qHyperpcovariance = qHypercovariance
            sigmafp = sigmaf       
            
            ### update policy paramter
            mu = qUmean
            gradMean = Bu.dot(invKuu).dot(mu)            
            
            theta += alpha*gradMean
        
        
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
