#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 14:07:57 2019

@author: mengmengcai
"""

import datetime as dt
import numpy as np
import pandas as pd
import scipy.stats as stats
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

def BS_d1(S, dt, r, sigma, K):
    return (np.log(S/K) + (r+sigma**2/2)*dt) / (sigma*np.sqrt(dt))

def BlackScholes_price(S, T, r, sigma, K, t=0):
    dt = T-t
    Phi = stats.norm(loc=0, scale=1).cdf
    d1 = BS_d1(S, dt, r, sigma, K)
    d2 = d1 - sigma*np.sqrt(dt)
    return S*Phi(d1) - K*np.exp(-r*dt)*Phi(d2)

def BS_delta(S, T, r, sigma, K, t=0):
    dt = T-t
    d1 = BS_d1(S, dt, r, sigma, K)
    Phi = stats.norm(loc=0, scale=1).cdf
    return Phi(d1)

#simulate the underlying price
def monte_carlo_paths(S_0, time_to_expiry, sigma, drift, seed, n_sims, n_timesteps):
    """
    Create random paths of a underlying following a browian geometric motion
    
    input:
    
    S_0 = Spot at t_0
    time_to_experiy = end of the timeseries (last observed time)
    sigma = the volatiltiy (sigma in the geometric brownian motion)
    drift = drift of the process
    n_sims = number of paths to generate
    n_timesteps = numbers of aquidistant time steps 
    
    return:
    
    a (n_timesteps x n_sims x 1) matrix
    """
    if seed > 0:
            np.random.seed(seed)
    stdnorm_random_variates = np.random.randn(n_sims, n_timesteps)
    S = S_0
    dt = time_to_expiry / stdnorm_random_variates.shape[1]
    r = drift
    # See Advanced Monte Carlo methods for barrier and related exotic options by Emmanuel Gobet
    S_T = S * np.cumprod(np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*stdnorm_random_variates), axis=1)
    return np.reshape(np.transpose(np.c_[np.ones(n_sims)*S_0, S_T]), (n_timesteps+1, n_sims, 1))

S_0 = 100
K = 100
r = 0
vol = 0.2
T = 1/12
timesteps = 30

# Train the model on the path of the risk neutral measure
paths_train = monte_carlo_paths(S_0, T, vol, r, 42, 500000, timesteps)

class RNNModel(object):
    def __init__(self, time_steps, batch_size, features, nodes = [62,46,46,1], name='model'):
        tf.reset_default_graph()
        self.batch_size = batch_size
        self.S_t_input = tf.placeholder(tf.float32, [batch_size, time_steps, features])
        self.K = tf.placeholder(tf.float32, batch_size)
        self.alpha = tf.placeholder(tf.float32)
        
        S_T = self.S_t_input[:, -1, 0]
        dS = self.S_t_input[:, 1:, 0] - self.S_t_input[:, :-1, :]
        
        #underlying spots except for the last one
        S_t = tf.unstack(self.S_t_input[:, :-1, :], axis=0)
        
        #build LSTM
        lstm = tf.contrib.rnn.multiRNNCell( [tf.contrib.rnn.LSTMCell(n) for n in nodes] )
        
        self.strategy, state = tf.nn.static_rnn( lstm, S_t, initial_state=
                                                lstm.zero_state(batch_size, tf.float32), 
                                                dtype=tf.float32)
        self.strategy = tf.reshape(self.strategy, (batch_size, time_steps-1))
        self.option = tf.maximum(S_T-self.K, 0)
        
        self.hedging_PNL = -self.option + tf.reduce_sum( dS*self.strategy, axis=1 )
        self.hedging_PNL_path = -self.option + dS*self.strategy
        
        # Calculate the CVaR for a given confidence level alpha
        # Take the 1-alpha largest losses (top 1-alpha negative PnLs) and calculate the mean
        CVaR, idx = tf.nn.top_k(-self.Hedging_PnL, tf.cast((1-self.alpha)*batch_size, tf.int32))
        CVaR = tf.reduce_mean(CVaR)
        self.train = tf.train.AdamOptimizer().minimize(CVaR)
        self.saver = tf.train.Saver()
        self.modelname = name

    def _execute_graph_batchwise(self, paths, strikes, riskaversion, sess, epochs=1, train_flag=False):
        sample_size = paths.shape[1]
        batch_size = self.batch_size
        idx = np.range(sample_size)
        start = dt.datetime.now()
        
        for epoch in range(epochs):
            #save the hedging PnL for each epoch
            pnls, strategies = [], []
            if train_flag:
                np.random.shuffle(idx)
            for i in range(int(sample_size/batch_size)):
                indices = idx[i*batch_size, (i+1)*batch_size]
                batch = paths[:, indices, :]
                if train_flag:
                    _, pnl, strategy = sess.run( [ self.train, self.hedging_PNL, self.strategy ],
                                                {self.S_t_input : batch, 
                                                 self.K         : strikes[indices],
                                                 self.alpha     : riskaversion
                                                }
                                                )
                else:
                    pnl, strategy = sess.run( [self.hedging_PNL, self.strategy],
                                                {self.S_t_input : batch, 
                                                 self.K         : strikes[indices],
                                                 self.alpha     : riskaversion
                                                }
                                                )
                pnls.append(pnl)
                strategies.append(strategy)
            #Calculate the option prive given the risk aversion level alpha
            CVaR = np.mean(-np.sort(np.concatenate(pnls))[:int((1-riskaversion)*sample_size)])
            if train_flag:
                if epoch % 10 == 0:
                    print('Time elapsed:', dt.datetime.now()-start)
                    print('Epoch', epoch, 'CVaR', CVaR)
                    self.saver.save(sess, r"/Users/desktop/coding/hedging" % self.modelname)
        self.saver.save(sess, r"/Users/desktop/coding/hedging" % self.modelname)
        return CVaR, np.concatenate(pnls), np.concatenate(strategies,axis=1)
    
    def training(self, paths, strikes, riskaversion, epochs, session, init=True):
        if init:
            sess.run(tf.global_variables_initializer())
        self._execute_graph_batchwise(paths, strikes, riskaversion, session, epochs, train_flag=True)
        
    def predict(self, paths, strikes, riskaversion, session):
        return self._execute_graph_batchwise(paths, strikes, riskaversion,session, 1, train_flag=False)

    def restore(self, session, checkpoint):
        self.saver.restore(session, checkpoint)    
        