# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:06:38 2021

@author: Xue
"""
import numpy as np
import matplotlib.pyplot as plt

class function():
    def __init__(self):
        self.par = 50
        self.dim = 10
        self.X_max = 10
        self.x = np.random.uniform(-self.X_max, self.X_max, [self.par, self.dim])

    def fitness(self, x):
        return np.sum(x**2, axis = 1)
    

def CSA(data, itermax=100):
    fl = 2
    AP = 0
    m = data.x.copy()
    fit = data.fitness(data.x)
    G_fit = np.min(fit)
    G, = np.where(G_fit==fit)
    Gbest = data.x[G[0]].copy()
    plt.figure()
    for _iter in range(itermax):
        for p in range(data.par):
            ri = np.random.randint(data.par)
            rp = np.random.rand()
            if rp > AP:
                data.x[p] += np.random.random_sample(data.x[p].shape) * fl * (m[ri]- data.x[p])
            else:
                data.x[p] = np.random.uniform(-data.X_max, data.X_max, data.x[p].shape)
        data.x[p] = np.clip(data.x[p], -data.X_max, data.X_max)
        new_fit = data.fitness(data.x)
        for p in range(data.par):
            if new_fit[p] < fit[p]:
                fit[p] = new_fit[p]
                m[p] = data.x[p].copy()
            else:
                data.x[p] = m[p].copy()
        new_G_fit = np.min(fit)
        if new_G_fit < G_fit:
            G_fit = new_G_fit
            G, = np.where(G_fit==fit)
            new_Gbest = data.x[G[0]].copy()
            Gbest = new_Gbest.copy()
        if G_fit == 0:
            print(_iter)
            return Gbest, G_fit
        if _iter % 5 == 0:
            # plot and show learning process
            #plt.cla()
            plt.title("dim 0 & 1")
            plt.xticks(np.arange(-12,12,2))
            plt.yticks(np.arange(-12,12,2))
            plt.scatter(data.x[0], data.x[1]) 
            th = np.arange(-3.14,3.14,0.01) 
            plt.plot(2*np.cos(th), 2*np.sin(th), color="black") 
            plt.plot(4*np.cos(th), 4*np.sin(th), color="black") 
            plt.plot(6*np.cos(th), 6*np.sin(th), color="black") 
            plt.plot(8*np.cos(th), 8*np.sin(th), color="black") 
            plt.plot(10*np.cos(th), 10*np.sin(th), color="black") 
        
            plt.pause(0.1)
    return Gbest, G_fit
            

if __name__ == "__main__":
    data = function()
    Gbest, G_fit = CSA(data, 100)
    #print(Gbest)
    print(G_fit)