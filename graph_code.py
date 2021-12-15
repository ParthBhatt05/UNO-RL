#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:35:04 2021

@author: fehringj
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('percent_win_data_2p.pkl', 'rb') as f:
    two = pickle.load(f)
with open('percent_win_data_3p.pkl', 'rb') as f:
    three = pickle.load(f)
with open('percent_win_data_4p.pkl', 'rb') as f:
    four = pickle.load(f)

print(two[-1], three[-1], four[-1])

#x = [0, 50000]
#y = [0.5, 0.5]

percent_x = np.linspace(1, (len(two)), len(two))
percent_y = two
percent_err = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))

percent_x2 = np.linspace(1, (len(three)), len(three))
percent_y2 = three
percent_err2 = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))

percent_x3 = np.linspace(1, (len(four)), len(four))
percent_y3 = four
percent_err3 = 1.96 * (np.std(percent_y) / np.sqrt(len(percent_y)))

fig1, ax1 = plt.subplots()

ax1.plot(two, label='two player')
ax1.plot(three, label='three player')
ax1.plot(four, label='four player')
#ax1.plot(x, y, '--')
ax1.set_ylim([0.6, 1])
ax1.fill_between(percent_x-1, (percent_y - percent_err), (percent_y + percent_err),
                 alpha=0.2)
ax1.fill_between(percent_x2-1, (percent_y2 - percent_err2), (percent_y2 + percent_err2),
                 alpha=0.2)
ax1.fill_between(percent_x3-1, (percent_y3 - percent_err3), (percent_y3 + percent_err3),
                 alpha=0.2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Percent Win')
ax1.legend(loc='best')
ax1.set_title('Percent Win for First Player')
fig1.savefig('dqn_multiple_player.png')