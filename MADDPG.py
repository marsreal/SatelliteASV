# -*- coding: utf-8 -*-
"""
Latest update on Wed Apr 27 16:15:43 2022

@author: mreal
"""



import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dtime

from SatelliteEnv import Env


env = Env()

np.random.seed(0)

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            
###############################            
class Buffer:
    def __init__(self,asv,actor_model,critic_model,target_actor,target_critic,actor_model2,target_actor2,actor_optimizer,critic_optimizer, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size
        # Num to start updating the networks
        self.min_size_buffer = 30000
        # Its tells us num of times record() was called.
        self.buffer_counter = 0
        
        self.actor_model = actor_model
        self.actor_model2 = actor_model2
        self.critic_model = critic_model
        self.target_actor = target_actor
        self.target_actor2 = target_actor2
        self.target_critic = target_critic
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions*2))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.state_full_buffer = np.zeros((self.buffer_capacity, num_states*2))
        self.next_state_full_buffer = np.zeros((self.buffer_capacity, num_states*2))

        self.asv = asv
    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        
        self.state_full_buffer[index] = obs_tuple[4]
        self.next_state_full_buffer[index] = obs_tuple[5]
        self.buffer_counter += 1
    def check_buffer_size(self):
        return self.buffer_counter >= self.batch_size and self.buffer_counter >= self.min_size_buffer
    
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,state_full_batch,next_state_full_batch
    ):
        # Training and updating Actor & Critic networks.

        if self.check_buffer_size()==False:
            return
        
        tacts = []
        with tf.GradientTape(persistent = True) as tape:
            if self.asv == 1:
                tacts.append(self.target_actor(next_state_full_batch[:,:7], training=True))
                tacts.append(self.target_actor2(next_state_full_batch[:,7:], training=True))
                target_actions = tf.concat(tacts,axis = 1)
            elif self.asv == 2:
                tacts.append(self.target_actor2(next_state_full_batch[:,:7], training=True))
                tacts.append(self.target_actor(next_state_full_batch[:,7:], training=True))
                target_actions = tf.concat(tacts,axis = 1)

            tc =  self.target_critic([next_state_full_batch, target_actions], training=True)
            
            y = reward_batch + gamma*tc
            critic_value = self.critic_model([state_full_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )
        acts = []
        
        with tf.GradientTape(persistent = True) as tape:
            if self.asv == 1:
                acts.append(self.actor_model(state_full_batch[:,:7], training=True))
                acts.append(self.actor_model2(state_full_batch[:,7:], training=True))
                actions = tf.concat(acts,axis = 1)
 
            elif self.asv == 2:
                acts.append(self.actor_model2(state_full_batch[:,:7], training=True))
                acts.append(self.actor_model(state_full_batch[:,7:], training=True))
                actions = tf.concat(acts,axis = 1)

            
            critic_value = self.critic_model([state_full_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)
        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        state_full_batch = tf.convert_to_tensor(self.state_full_buffer[batch_indices])
        next_state_full_batch = tf.convert_to_tensor(self.next_state_full_buffer[batch_indices])
        

        self.update(state_batch, action_batch, reward_batch, next_state_batch,state_full_batch,next_state_full_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))            
            
###########################3            
def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(512, activation="relu")(out)
    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)

    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states*2,))
    state_out = layers.Dense(32, activation="relu")(state_input)
    state_out = layers.Dense(64, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions*2,))
    action_out = layers.Dense(64, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(512, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model            


######################################33           

def policy(state, noise_object,actor_model):
  
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)
    
    return [np.squeeze(legal_action)]

##################################
num_states = 7
num_actions = 2

upper_bound = 1
lower_bound = -1

std_dev = 0.2
ou_noise1 = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
ou_noise2 = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))
actor_model1 = get_actor()
critic_model1 = get_critic()

target_actor1 = get_actor()
target_critic1 = get_critic()

# Making the weights equal initially
target_actor1.set_weights(actor_model1.get_weights())
target_critic1.set_weights(critic_model1.get_weights())

actor_model2 = get_actor()
critic_model2 = get_critic()

target_actor2 = get_actor()
target_critic2 = get_critic()

# Making the weights equal initially
target_actor2.set_weights(actor_model2.get_weights())
target_critic2.set_weights(critic_model2.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.0006
actor_lr = 0.0005

actor_lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate = actor_lr,decay_steps=50000,decay_rate=0.97,staircase=True)
critic_lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate = critic_lr,decay_steps=50000,decay_rate=0.97,staircase=True)
        
critic_optimizer1 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
actor_optimizer1 = tf.keras.optimizers.Adam(learning_rate=actor_lr)
critic_optimizer2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
actor_optimizer2 = tf.keras.optimizers.Adam(learning_rate=actor_lr)

total_episodes = 650
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.001

buffer1 = Buffer(1,actor_model1,critic_model1,target_actor1,target_critic1,actor_model2,target_actor2,actor_optimizer1,critic_optimizer1,100000, 64)
buffer2 = Buffer(2,actor_model2,critic_model2,target_actor2,target_critic2,actor_model1,target_actor1,actor_optimizer2,critic_optimizer2,100000, 64)
##############################
# To store reward history of each episode
ep_reward_list1 = []
avg_reward_list1 = []
ep_reward_list2 = []
avg_reward_list2 = []

phase = 0    #what phase of difficulty is going to be used
n_eps = 0   #counter of episodes for storage

for ep in range(total_episodes):

    prev_state = env.reset(phase)
    episodic_reward1 = 0
    episodic_reward2 = 0
    
    #To store each individual reward function
    epindv1 = np.zeros((2,))
    epindv2 = np.zeros((2,))
    epindv3 = np.zeros((2,))
    epindv4 = np.zeros((2,))
    epindv5 = np.zeros((2,))
    epindv6 = np.zeros((2,))
     
    while True:

        print('Episode: ', ep)
        tf_prev_state1 = tf.expand_dims(tf.convert_to_tensor(prev_state[0]), 0)
        tf_prev_state2 = tf.expand_dims(tf.convert_to_tensor(prev_state[1]), 0)
        
        #Produce actions per each model
        action1 = policy(tf_prev_state1, ou_noise1,actor_model1)
        action2 = policy(tf_prev_state2, ou_noise2,actor_model2)

        # Recieve state and reward from environment.
        reward,state, done,indv_rw= env.step(action1,action2)

        state_full = np.concatenate(prev_state)   
        next_state_full = np.concatenate(state)
        actions = []
        actions.append(action1)
        actions.append(action2)
        action_full = np.concatenate(actions,axis = 1)

        buffer1.record((prev_state[0], action_full, reward[0], state[0],state_full,next_state_full))
        buffer2.record((prev_state[1], action_full, reward[1], state[1],state_full,next_state_full))
        
        episodic_reward1 += reward[0]
        episodic_reward2 += reward[1]
        for a in range(2):
            epindv1[a] += indv_rw[a][0]
            epindv2[a] += indv_rw[a][1]
            epindv3[a] += indv_rw[a][2]
            epindv4[a] += indv_rw[a][3]
            epindv5[a] += indv_rw[a][4]
            epindv6[a] += indv_rw[a][5]
            
        if buffer1.check_buffer_size():
            buffer1.learn()
            update_target(target_actor1.variables, actor_model1.variables, tau)
            update_target(target_critic1.variables, critic_model1.variables, tau)
            
        if buffer2.check_buffer_size():
            buffer2.learn()
            update_target(target_actor2.variables, actor_model2.variables, tau)
            update_target(target_critic2.variables, critic_model2.variables, tau)
            
        # End this episode when `done` is True
        if done:
            #Store reward info in the environment
            env.highscore1(episodic_reward1,epindv1[0],epindv2[0],epindv3[0],epindv4[0],epindv5[0],epindv6[0])
            env.highscore2(episodic_reward2,epindv1[1],epindv2[1],epindv3[1],epindv4[1],epindv5[1],epindv6[1])
            break

        prev_state[0] = state[0]
        prev_state[1] = state[1]
        
        ##To advance phase:
        # if ep>300:
        #     phase = 2
        # elif ep>500:
        #     phase = 3
        
    ep_reward_list1.append(episodic_reward1)
    # Mean of last 40 episodes
    avg_reward1 = np.mean(ep_reward_list1[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward1))
    avg_reward_list1.append(avg_reward1)
    
    ep_reward_list2.append(episodic_reward2)

    # Mean of last 40 episodes
    avg_reward2 = np.mean(ep_reward_list2[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward2))
    avg_reward_list2.append(avg_reward2)
    n_eps +=1
    
    #In case the average reward is higher than -175, the models are stored
    if avg_reward1>-175 and avg_reward2>-175 and n_eps>50:
                timenow = dtime.now()
                tstamp = timenow.strftime('_%d_%m_%M_%S')
                
                filename1 = 'Mod_maddpg1''_'+str(ep)+tstamp+'.h5'
                actor_model1.save(filename1)
                filename2 = 'Mod_maddpg2''_'+str(ep)+tstamp+'.h5'
                actor_model2.save(filename2)
                n_eps = 0
                
    #Once the training has surpassed a number of episodes and both agents get a high reward the models are stored
    if ep>(total_episodes-100) and episodic_reward1>-150 and episodic_reward2>-150:
        timenow = dtime.now()
        tstamp = timenow.strftime('_%d_%m_%M_%S')
        filename1 = 'Mod_maddpg1''_'+str(ep)+tstamp+'.h5'
        actor_model1.save(filename1)
        filename2 = 'Mod_maddpg2''_'+str(ep)+tstamp+'.h5'
        actor_model2.save(filename2)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(ep_reward_list1)
plt.plot(avg_reward_list1)
plt.plot(avg_reward_list2)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.title('ASV 1')
plt.show()  

plt.plot(ep_reward_list2)
plt.plot(avg_reward_list2)
plt.plot(avg_reward_list1)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.title('ASV 2')
plt.show()        

#Save the final model
filename1 = 'Model_maddpg1.h5'
actor_model1.save(filename1)  
filename2 = 'Model_maddpg2.h5'
actor_model2.save(filename2)      

            
            
            
            
            
            
