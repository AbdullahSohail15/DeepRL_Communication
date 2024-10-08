# -*- coding: utf-8 -*-
"""
Modified version based on the code by mengxiaomao for the paper
https://arxiv.org/abs/1901.07159
"""
import scipy
from scipy import special
import numpy as np
from scipy.special import lambertw
import math

'''Reinforcement Learning Environment for the non-linear energy harvesting model'''

dtype = np.float32

def get_practical_power(P_n,hn_0,theeta,alpha,beta):
    P_prac=theeta*((math.exp(P_n*hn_0*alpha)-1)/(math.exp(P_n*hn_0*alpha)+math.exp(alpha*beta)))
    return P_prac

class Env_cellular():

    def __init__(self, MAX_EP_STEPS, s_dim, location_vector, location_GF,K, Pn,fading_n, fading_0,theeta,alpha,beta):
        self.emax = 0.1  # battery capacity
        #self.P0 = P0
        self.K = K
        self.T =1
        self.eta =0.7
        self.Pn = Pn # grant-based user's power
        self.Pmax = 0.1
        self.alpha = alpha  
        self.beta = beta
        self.theeta = theeta


        BW = 10**6 # 10MHz
        sigma2_dbm = -170 + 10 * np.log10(BW) #  Thermal noise in dBm
        #sigma2_dbm = -94
        self.noise = 10 ** ((sigma2_dbm - 30) / 10)
        self.Pn = self.Pn

        #self.P0n = 10 #grant free user's power
        self.s_dim = s_dim

        self.MAX_EP_STEPS = MAX_EP_STEPS


        distance_GF = np.sqrt(np.sum((location_vector-location_GF)**2, axis=1))
        distance_GB = np.sqrt(np.sum((location_vector)**2, axis=1))

        distance = np.matrix.transpose(np.array([distance_GF, distance_GB]))
        distance = np.maximum(distance, np.ones((self.K,2)))
        PL_alpha = 3
        PL = 1/distance**PL_alpha/(10**3.17)
        self.hn = np.multiply( PL, fading_n)
        self.hn=self.hn*5.7713922
        distance_GF0 = np.sqrt(np.sum( location_GF ** 2, axis=1))
        distance0 = np.maximum(distance_GF0, 1)
        PL0 = 1/(distance0 ** PL_alpha)/(10**3.17)
        self.h0 = fading_0*PL0
        print(f"{self.hn} and {self.h0}")
        print(f"{fading_n} and {fading_0}")

        self.channel_sequence = np.zeros(( self.MAX_EP_STEPS,2))
        for i in range(self.MAX_EP_STEPS):
            id_index = i % self.K
            self.channel_sequence[i,:] = self.hn[id_index,:]

    def step(self, action, state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise #state[0,2]/self.noise
        En = state[0,-1]
        Pn_prac=get_practical_power(self.Pn,hn0,self.theeta,self.alpha,self.beta)

        En_bar = action*min(self.emax-En, self.T*Pn_prac) - (1-action)*min(En,self.T*self.Pmax)

        mu1 = Pn_prac*h0/(1+self.Pn*hn)
        mu2 = En_bar*h0/(self.T*(1+self.Pn*hn))
        wx0 = np.real(lambertw(math.exp(-1)*(mu1-1), k=0))
        alphaxx = (mu1-mu2)/(math.exp(wx0+1) - 1 + mu1)
        alpha01 = 1 - (En+En_bar)/(self.T*Pn_prac)
        alpha02 = (self.T*Pn_prac-En_bar)\
                  /(self.T*Pn_prac + self.T*self.Pmax)
        alphax2 = max(alpha01, alpha02)
        alphan = min(1, max(alphaxx,alphax2))
        if En_bar == self.T * Pn_prac:
            P0n=0
            reward = [0]  # remark in the paper
        elif alphan ==0:#<= 0.00000001:
            P0n=0
            reward = [0]
        else:
            P0n = (1-alphan)*Pn_prac/alphan - En_bar/alphan/self.T
            reward = alphan*np.log(1 + P0n*h0/(1 +self.Pn*hn))

        
        if math.isnan(reward[0]):
            print(f"start---------------------------------------")
            print(f"action is{action} ")
            print(f"finallly alphan is {alphan}")
            print(f"end-----------------------------------")
            reward = 0
        


        batter_new = min(self.emax, En + En_bar)


        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        state_next.append(float(batter_new))
        state_next = np.reshape(state_next, (1, self.s_dim))
        done=False
        energy_harvested=(1-alphan)*self.T*Pn_prac

        return reward, state_next, done, alphan

    def step_greedy(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]

        alphan = min(1, En/self.T/self.Pmax)
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 +self.Pn*hn))

        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)

        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
        
        done=False

        return reward, state_next, done

    def step_random(self,   state, j):
        hn = state[0,0]/self.noise
        hn0 = state[0,1]
        h0 = self.h0/self.noise
        En = state[0,-1]

        alphan = np.random.uniform(0, min(1, En/self.T/self.Pmax))
        if alphan==0:
            reward = 0
        else:
            reward = alphan*np.log(1 + self.Pmax*h0/(1 +self.Pn*hn))


        if math.isnan(reward):
            print(f"alpha is {alphan}  ")
            print(f"{En}")
        batter_new = min(self.emax, En -alphan*self.T*self.Pmax +(1-alphan)*self.T*self.eta*self.Pn*hn0)

        state_next = self.channel_sequence[(j+1) % self.K, :].tolist()
        state_next.append(batter_new)
        state_next = np.reshape(state_next, (1, self.s_dim))
       

        done=False

        return reward, state_next, done

    def reset(self):
        batter_ini = self.emax
        return batter_ini

print("Successfully Compiled")