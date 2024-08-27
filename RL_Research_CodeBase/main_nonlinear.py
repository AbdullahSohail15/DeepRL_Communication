#Imports
import numpy as np
import torch
import time
from enviroment_nonlinear import Env_cellular as env
import matplotlib.pyplot as plt
from PER_DDPG import DDPG as PER_DDPG
import PER_buffer
import utils
from td3_torch import Agent as TD3_AGENT
from ddpg_torch import Agent as DDPG_AGENT
from PPO import PPO
from CER_Buffer import CompositeReplayBuffer
from CER_DDPG import CER_DDPG

'''Main file for the non-linear energy harvesting model'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#####################  hyper parameters  ####################
## Pn = 0.5, 0.676, 0.84, 1, 1.2
Pn = 1.2
K=2
THEETA=0.02
ALPHA=6400
BETA=0.003
MAX_EPISODES = 150
MAX_EP_STEPS = 100
LR_A = 0.002  # learning rate for actor
LR_C = 0.004    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

#Output format
#True = training and testing results
#False = only testing results are printed
verbose = True

s_dim = 3# dimsion of states
a_dim = 1# dimension of action
a_bound = 1 #bound of action
state_am = 1000

# Location of Primary Devices
location_vector = np.array([[0, 1],[0,1000]]) # K=2
#location_vector = np.array([[0, 1],[0,334],[0,667],[0,1000]])                                                #(K=4)
#location_vector = np.array([[0, 1],[0,200.8],[0,400.6],[0,600.4],[0,800.2],[0,1000]])                        #K=6
#location_vector = np.array([[0, 1],[0,143.714],[0,286.428],[0,429.142],[0,571.856],[0,714.57],[0,857.284],[0,1000]]) #K=8
#location_vector = np.array([[0, 1],[0,112],[0,223],[0,334],[0,445],[0,556],[0,667],[0,778],[0,889],[0,1000]])        #K=10

# Location of RCSD
location_GF = np.array([[1,1]])# np.ones((1, 2))

##### fading for Primary  users
hnx1 = np.random.randn(K, 2)
hnx2 = np.random.randn(K, 2)
fading_n = 1#hnx1 ** 2 + hnx2 ** 2

#### fading for RCSD
h0x1 = np.random.randn(1, 1)
h0x2 = np.random.randn(1, 1)
fading_0 = 1#h0x1[0,0] ** 2 + h0x2[0,0] ** 2

myenv = env( MAX_EP_STEPS, s_dim, location_vector,location_GF,K,Pn, fading_n, fading_0,THEETA,ALPHA,BETA)

#Experimental Parameters
#Set random seed number
seed = 0



#-----Set seeds-------
torch.manual_seed(seed)
np.random.seed(seed)
#--------------------

#---------------------------------Initializing DDPG POLICY---------------------------------------------------------------------------------------
ddpg_agent=DDPG_AGENT(LR_A,LR_C,s_dim,TAU,a_bound,GAMMA,n_actions=1,max_size=MEMORY_CAPACITY,batch_size=BATCH_SIZE)
#------------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------Initializing PER-DDPG POLICY-----------------------------------------------------
#PER parameters
prioritized_replay_alpha=0.6
prioritized_replay_beta0=0.4
prioritized_replay_beta_iters=None
prioritized_replay_eps=10000
#Create DDPG policy_per_ddpg
policy_per_ddpg = PER_DDPG(s_dim, a_dim, a_bound)
#If we are not doing prioritized experience replay
#Then we use my implementation of the uniform replay buffe
replay_buffer = PER_buffer.PrioritizedReplayBuffer(MEMORY_CAPACITY, prioritized_replay_alpha)
if prioritized_replay_beta_iters is None:
    prioritized_replay_beta_iters = MAX_EP_STEPS*MAX_EPISODES
#Create annealing schedule
beta_schedule = utils.LinearSchedule(prioritized_replay_beta_iters, initial_p=prioritized_replay_beta0, final_p=1.0)
#--------------------------------------------------------------------------------------------------------------------

#---------------------------------Initializing CER-DDPG POLICY----------------------------------------------------
initial_alpha=0
final_alpha=0.5
alpha_schedule = utils.LinearSchedule(MAX_EPISODES*MAX_EP_STEPS, initial_p=initial_alpha, final_p=final_alpha)
eeta=0.5
replay_buffer_cer=CompositeReplayBuffer(MEMORY_CAPACITY)
cer_agent=CER_DDPG(s_dim, a_dim, a_bound,eeta)
#--------------------------------------------------------------------------------------------------------------------

#---------------------------------Initializing TD3 POLICY--------------------------------------------------------------------------------------
td3_agent=TD3_AGENT(LR_A,LR_C,s_dim,TAU,a_bound,GAMMA,update_actor_interval=2,n_actions=1,max_size=MEMORY_CAPACITY,batch_size=BATCH_SIZE)
#---------------------------------------------------------------------------------------------------------------------------------------------

#---------------------------------initialize PPO POLICY--------------------------------------------------------------------------------------
has_continuous_action_space = True  # continuous action space; else discret
action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
action_std_decay_freq = int(2.5e6)  # action_std decay frequency (in num timesteps)
update_timestep = MAX_EP_STEPS * 3     # update policy every n timesteps
K_epochs = 40            # update policy for K epochs in one PPO update
eps_clip = 0.2          # clip parameter for PPO

ppo_agent = PPO(s_dim, a_dim, LR_A,LR_C, GAMMA, K_epochs, eps_clip, has_continuous_action_space, action_std)
#--------------------------------------------------------------------------------------------------------------------------------------------


var = 1
total_time = 1
episode_t=MAX_EP_STEPS
episode = 0
s_traj_ddpg = []
s_traj_perddpg=[]
s_traj_td3 = []
s_traj_ppo=[]
s_traj_cerddpg=[]
t_0 = time.time()
ep_rewardall_per_ddpg=[]
ep_rewardall_td3=[]
ep_rewardall_ddpg=[]
ep_rewardall_cer_ddpg=[]
ep_rewardall_ppo=[]
energy_harvested=[]
energy_harvested_per=[]
energy_harvested_td3=[]
energy_harvested_cer=[]
energy_harvested_ppo=[]
while episode<=MAX_EPISODES:
    batter_ini = myenv.reset()
    s_perddpg = myenv.channel_sequence[episode%myenv.K,:].tolist() #the current GB user, 2 element [GB-GF, GB-BS] #s.append(myenv.h0)
    s_perddpg.append(batter_ini)
    s_perddpg = np.reshape(s_perddpg,(1,s_dim))
    s_perddpg = s_perddpg*state_am #amplify the state
    s_td3 = s_perddpg
    s_ddpg=s_perddpg
    s_ppo=s_perddpg
    s_cerddpg=s_perddpg
    episode_r_per_ddpg=0
    episode_r_td3=0
    episode_r_ddpg=0
    episode_r_ppo=0
    episode_r_cer_ddpg=0
    for episode_t in range(MAX_EP_STEPS):
#-----------------States and Actions Transitions DDPG----------------------------------------------------------------
        #Given current state, get action
        a = ddpg_agent.choose_action(np.array(s_ddpg))
        #Apply exploration noise to action
        a = np.clip(np.random.normal(a, var), 0, 1)
        #Using action, take step in environment, observe new state, reward and episode status
        r, s_ddpg_, done,e_h = myenv.step(a,s_ddpg/state_am,episode_t)
        energy_harvested.append(e_h)
        s_ddpg_ = s_ddpg_*state_am
        s_traj_ddpg.append(s_ddpg_)
        episode_r_ddpg += r
        # Store data in replay buffer
        ddpg_agent.remember(s_ddpg[0], a, r, s_ddpg_[0], 0)
#---------------------------------------------------------------------------------------------------------------------

#-----------------States and Actions Transitions-CER DDPG----------------------------------------------------------------
        #Given current state, get action
        a = cer_agent.get_action(np.array(s_cerddpg))
        #Apply exploration noise to action
        a = np.clip(np.random.normal(a, var), 0, 1)
        #Using action, take step in environment, observe new state, reward and episode status
        r, s_cerddpg_, done,e_h_cer = myenv.step(a,s_cerddpg/state_am,episode_t)
        energy_harvested_cer.append(e_h_cer)
        s_cerddpg_ = s_cerddpg_*state_am
        s_traj_cerddpg.append(s_cerddpg_)
        episode_r_cer_ddpg += r
        # Store data in replay buffer
        replay_buffer_cer.add(s_cerddpg[0], a, r, s_cerddpg_[0], 0)
#---------------------------------------------------------------------------------------------------------------

#-----------------States and Actions Transitions-PER DDPG-----------------------------------------------------------------
        #Given current state, get action
        a = policy_per_ddpg.get_action(np.array(s_perddpg))
        #Apply exploration noise to action
        a = np.clip(np.random.normal(a, var), 0, 1)
        #Using action, take step in environment, observe new state, reward and episode status
        r, s_perddpg_, done,e_h_per = myenv.step(a,s_perddpg/state_am,episode_t)
        energy_harvested_per.append(e_h_per)
        s_perddpg_ = s_perddpg_*state_am
        s_traj_perddpg.append(s_perddpg_)
        episode_r_per_ddpg += r
        # Store data in replay buffer
        replay_buffer.add(s_perddpg[0], a, r, s_perddpg_[0], 0)
#--------------------------------------------------------------------------------------------------------------------------

#----------------States and Actions Transitions-TD3------------------------------------------------------------------------
        #Given current state, get action
        a = td3_agent.choose_action(np.array(s_td3))
        #Apply exploration noise to action
        a = np.clip(np.random.normal(a, var), 0, 1)
        #Using action, take step in environment, observe new state, reward and episode status
        r, s_td3_, done,e_h_td3 = myenv.step(a,s_td3/state_am,episode_t)
        energy_harvested_td3.append(e_h_td3)
        s_td3_ = s_td3_*state_am
        s_traj_td3.append(s_td3_)
        episode_r_td3 += r
        # Store data in replay buffer
        td3_agent.remember(s_td3[0], a, r, s_td3_[0], 0)
#---------------------------------------------------------------------------------------------------------------------------
#-----------------States and Actions Transitions-PPO------------------------------------------------------------------------
        a = ppo_agent.select_action(s_ppo)
        a = np.clip(np.random.normal(a, var), 0, 1)
        r, s_ppo_, done,e_h_ppo= myenv.step(a,s_ppo/state_am,episode_t)
        energy_harvested_ppo.append(e_h_ppo)
        s_ppo_ = s_ppo_*state_am
        s_traj_ppo.append(s_ppo_)
        episode_r_ppo += r
        ppo_agent.buffer.rewards.append(r)
        if episode_t==MAX_EP_STEPS-1:
              ppo_agent.buffer.is_terminals.append(1)
        else:
              ppo_agent.buffer.is_terminals.append(0)
#---------------------------------------------------------------------------------------------------------------------------

        if var > 0.1:
                    var *= .9998

#----------------DDPG TRAINING---------------------------------------------------------------------------------
        ddpg_agent.learn()
#--------------------------------------------------------------------------------------------------------------

#----------------PER-DDPG ALGORITHM TRAINING-----------------------------------------------------------------------------
        beta_value = 0
        beta_value = beta_schedule.value(total_time)
        #print("---TRAINING PER-DDPG---")
        policy_per_ddpg.train(replay_buffer, True, beta_value, prioritized_replay_eps,BATCH_SIZE, GAMMA,TAU)
#---------------------------------------------------------------------------------------------------------------

#----------------CER-DDPG ALGORITHM TRAINING-----------------------------------------------------------------------------
        alpha=0
        alpha=alpha_schedule.value(total_time)
        #print("---TRAINING PER-DDPG---")
        cer_agent.train(replay_buffer_cer, True, beta_value, prioritized_replay_eps,alpha,BATCH_SIZE, GAMMA,TAU)
#------------------------------------------------------------------------------------------------------------------------

#----------------TD3 TRAINING-----------------------------------------------------------------------------
        td3_agent.learn()
#---------------------------------------------------------------------------------------------------------

#----------------PPO-Training-----------------------------------------------------------------------------
        if total_time % update_timestep == 0:
            ppo_agent.update()
        if has_continuous_action_space and total_time % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
#---------------------------------------------------------------------------------------------------------
        s_ddpg=s_ddpg_
        s_perddpg=s_perddpg_
        s_td3=s_td3_
        s_ppo=s_ppo_
        s_cerddpg=s_cerddpg_
        total_time+=1

        if episode_t==MAX_EP_STEPS-1:
            if verbose:
                print("Episodes: "+str(episode)+" Episode Reward DDPG: "+str(episode_r_ddpg)+ " Episode Reward PER DDPG: "+
                      str(episode_r_per_ddpg)+ " Episode Reward CER DDPG: "+
                      str(episode_r_cer_ddpg)+ " Episode Reward TD3: "+str(episode_r_td3)+
                      " Episode Reward PPO: "+str(episode_r_ppo)+" Runtime: "+str(int(time.time() - t_0)))
                #print("===========================================")
                #print(f"Harvested Energy: {e_h}")
                
            ep_reward_perddpg = np.reshape(episode_r_per_ddpg/MAX_EP_STEPS, (1,))
            ep_rewardall_per_ddpg.append(float(ep_reward_perddpg))
            ep_reward_cerddpg=np.reshape(episode_r_cer_ddpg/MAX_EP_STEPS, (1,))
            ep_rewardall_cer_ddpg.append(float(ep_reward_cerddpg))
            ep_reward_ddpg = np.reshape(episode_r_ddpg/MAX_EP_STEPS, (1,))
            ep_rewardall_ddpg.append(float(ep_reward_ddpg))
            ep_reward_td3 = np.reshape(episode_r_td3/MAX_EP_STEPS, (1,))
            ep_rewardall_td3.append(float(ep_reward_td3))
            ep_reward_ppo = np.reshape(episode_r_ppo/MAX_EP_STEPS, (1,))
            ep_rewardall_ppo.append(float(ep_reward_ppo))
            episode+=1
'''            
rewards_save=[ep_rewardall_per_ddpg,ep_rewardall_cer_ddpg,ep_rewardall_ddpg,ep_rewardall_td3,ep_rewardall_ppo]
file = open('non_linear_2.txt','a')
for item in rewards_save:
      file.write(str(item))
      file.write("\n")
file.close()
'''
# Averaging results over 150 episodes
harvested_energy=[sum(energy_harvested_per)/len(energy_harvested_per),
                  sum(energy_harvested_cer)/len(energy_harvested_cer),
                  sum(energy_harvested)/len(energy_harvested),
                  sum(energy_harvested_td3)/len(energy_harvested_td3),
                  sum(energy_harvested_ppo)/len(energy_harvested_ppo)]

file=open('non_linear_alphan.txt','a')
for item in harvested_energy:
      print("Average alphan: ", item)
      file.write(str(item))
      file.write("\n")
file.close()

print("=================================================================================")
print(f"Average Power Transmitted (DDPG): {sum(energy_harvested)/len(energy_harvested)}")
print(f"Average Power Transmitted (PER-DDPG): {sum(energy_harvested_per)/len(energy_harvested_per)}")
print(f"Average Power Transmitted (TD3): {sum(energy_harvested_td3)/len(energy_harvested_td3)}")
print(f"Average Power Transmitted (CER-DDPG): {sum(energy_harvested_cer)/len(energy_harvested_cer)}")
print(f"Average Power Transmitted (PPO): {sum(energy_harvested_ppo)/len(energy_harvested_ppo)}")
print("==================================================================================")
print(f"{myenv.hn} and {myenv.h0}")
plt.plot(ep_rewardall_cer_ddpg,"v-", label='CER DDPG: rewards')
plt.plot(ep_rewardall_ddpg, "o--", label='DDPG: rewards')
plt.plot(ep_rewardall_per_ddpg, "^-", label='PER DDPG: rewards')
plt.plot(ep_rewardall_td3, "+:", label='TD3: rewards')
plt.plot(ep_rewardall_ppo,"x--", label='PPO: rewards')
plt.xlabel("Episode")
plt.ylabel(" Epsiodic Reward -  Data Rate (NPCU)")
plt.legend()
plt.show()
