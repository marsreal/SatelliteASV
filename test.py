# -*- coding: utf-8 -*-
"""
Latest update on Wed Apr 27 17:36:53 2022

@author: mreal
"""

import tensorflow as tf
import numpy as np
from testEnv import Env

env = Env()

def policy(state,actor_model):
    sampled_actions = actor_model.predict(state)

    return [np.squeeze(sampled_actions)]


def mission(act_model0,act_model1,env):
    #initialization of simulation
    prev_state1 = env.reset(1)
    prev_state0 = env.reset(0)


    while True:

        tf_prev_state0 = tf.expand_dims(tf.convert_to_tensor(prev_state0), 0)
        
        action0 = policy(tf_prev_state0,act_model0)

        tf_prev_state1 = tf.expand_dims(tf.convert_to_tensor(prev_state1), 0)
        
        action1 = policy(tf_prev_state1,act_model1)

        # Recieve state from environment.

        state0 = env.Bot1(action0)

        state1 = env.Bot2(action1)
        
        done= env.updatestate()

        if done:
            stats = env.Statistics()
            break

        prev_state0 = state0
        prev_state1 = state1
    return stats
        
if __name__ == '__main__':

    #load models
    model1 = tf.keras.models.load_model('Model_ddpg1_pec.h5',compile = True)
    model2 = tf.keras.models.load_model('Model_ddpg2_pec.h5',compile = True)

    stats = mission(model1,model2,env)
