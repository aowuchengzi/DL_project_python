# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 10:25:55 2018

@author: 15339
"""

import numpy as np
#%%

mask = array([[[0., 1.],
               [0., 1.],
               [0., 1.]],

              [[0., 1.],
               [0., 1.],
               [0., 1.]],

              [[0., 1.],
               [0., 1.],
               [0., 1.]]])
new_mask = np.zeros((3, 3) + (2, ))
new_mask
mask == 0
mask[1,1]=3
#%%
for i in range(2):
   
    new_mask[mask == i,i] = 1
new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) 
mask = new_mask
