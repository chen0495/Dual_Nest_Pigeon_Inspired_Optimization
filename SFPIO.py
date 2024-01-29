import random

from .Population import Pops
from .BasePIO import BasePIO
import numpy as np
import scipy.special as sc_special
import time
import math
import copy

class SFPIO(BasePIO):
    def __init__(self, params, **kwargs):
        '''
        :param kwargs: dict
        such as:
            {
            "popSize" : 40
            "vardim": 50
            "bound" : [[l_1,l_2,...,l_n],[u_1,u_2,...,u_n]] # and n=vardim
            "Maxiter" : 1000
            "step" : in map factor, 60
            "k" : in map factor, 4
            "g" : guess step in landmark factor
            "init_type" : 0 for oringin、1 for latin
            "personal_type": True with Personal_Best updated
            "func" : function of calculating fitness
            "M" : matrix of input,E
            "S" : shuffle of input,0
            }
        '''
        self.popSize, self.vardim, self.bound = 40, 50, np.tile([[-100], [100]], 50)
        self.func, self.M, self.S = None, np.eye(self.vardim),np.zeros(self.vardim)
        self.init_type, self.personal_type = 0, False
        self.Maxiter = 1000
        self.step = 60
        self.k = 16
        self.c = 0.5
        self.percent = 0.8

        self.__set_keyword_arguments(params)
        self.__set_keyword_arguments(kwargs)

        self.Nc1 = int(self.percent * self.Maxiter)
        self.Nc2 = self.Maxiter - self.Nc1
        self.Pops = Pops(params=params,personal_type=True,init_type=1)
        self.half = self.popSize
        #print("xxx==",self.Pops.fitness)

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():

            setattr(self, key, value)

    def choise(self,n=2):
        #half = int(self.popSize*0.2)+2
        minf = np.min(self.Pops.fitness)
        idx = 1/(self.Pops.fitness-minf+1e-10)
        p = idx/np.sum(idx)
        return np.random.choice(self.popSize, size=self.popSize, p=p, replace=True)

    def choise_3(self,n):
        rank = np.argsort(self.Pops.fitness)
        p = np.ones(self.popSize) * self.c
        p = p ** rank
        p /= p.sum()
        return np.random.choice(self.popSize, size=n, p=p, replace=True)

    def move_part_one(self,epoch):
        t = 1/(1+np.exp(self.k*(epoch/self.Nc1))) #此处可偏移
        # choised = self.choise_2(self.popSize)#np.random.randint(0,20,size=self.popSize)
        choised = self.choise_3(self.popSize)

        r=np.random.random((self.popSize,1))* 3 - 1
        mid_point = r*self.Pops.personal_best_position[choised]+(1-r)*self.Pops.personal_best_position
        self.Pops.v = 0.5*(self.bound[1]-self.bound[0])*np.random.normal(0,t,(self.popSize,self.vardim))
        self.Pops.pop= self.Pops.v + mid_point

#         r = np.random.random(self.popSize)
#         idx = r<0.5
#         self.Pops.pop[idx]= self.Pops.v[idx] + mid_point[idx]
#         idx = r>=0.5
#         self.Pops.pop[idx] = mid_point[idx]


    def move_part_two(self, epoch):
        t = 1/(1+np.exp(self.k*(epoch/self.Nc2-0.5)))
        self.Pops.v = (self.Pops.personal_best_position - self.Pops.global_best_position)*np.random.normal(0,t,(self.popSize,self.vardim))
        self.Pops.pop = self.Pops.global_best_position+self.Pops.v

    # def updated_worst(self):
    #     idx = self.Pops.fitness != self.Pops.personal_best_fitness #未更新
    #     if np.any(idx)==0:
    #         return
    #     min_index = np.argmin(self.Pops.fitness[idx]) #未更新中最小
    #     max_index = np.argmax(self.Pops.personal_best_fitness) #最大个体历史最优
    #     if (self.Pops.fitness[idx][min_index]<self.Pops.personal_best_fitness[max_index]):
    #         self.Pops.personal_best_position[max_index] = self.Pops.pop[idx][min_index]
    #         self.Pops.personal_best_fitness[max_index] = self.Pops.fitness[idx][min_index]
    def solve(self):

        print("***{}***".format(self.__class__.__name__))
        st = time.time()
        #self.Pops.absorb_p = 0.5
        for t in range(self.Nc1):           #指南针算子
            self.move_part_one(t)            #更新种群位置
            self.Pops.updated_2()             #更新种群状态


        print("Part.{},  fitness = {}".format(t,self.Pops.global_best_fitness))

        for t in range(self.Nc2):
            self.move_part_two(t)
            self.Pops.updated_2()

        et = time.time()
        print("run_time={}s, best_fitness = {}".format(et-st,self.Pops.global_best_fitness))

    # def solve(self):
    #     x = np.zeros((self.popSize,self.Maxiter))
    #     y = np.zeros((self.popSize,self.Maxiter))
    #     print("***{}***".format(self.__class__.__name__))
    #     st = time.time()
    #     #self.Pops.absorb_p = 0.5
    #     for t in range(self.Nc1):           #指南针算子
    #         self.move_part_one(t)            #更新种群位置
    #         self.Pops.updated_2()             #更新种群状态
    #         x[:,t]=self.Pops.pop[:,0]
    #         y[:,t] = self.Pops.pop[:, 1]
    #
    #
    #     print("Part.{},  fitness = {}".format(t,self.Pops.global_best_fitness))
    #
    #     for t in range(self.Nc2):
    #         self.move_part_two(t)
    #         self.Pops.updated_2()
    #         x[:, t+self.Nc1] = self.Pops.pop[:, 0]
    #         y[:, t+self.Nc1] = self.Pops.pop[:, 1]
    #
    #     et = time.time()
    #     print("run_time={}s, best_fitness = {}".format(et-st,self.Pops.global_best_fitness))
    #
    #     return x,y


if __name__ == "__main__":
    import TempTestFunction as Func

    params = {
        "popSize": 10,
        "vardim": 2,
        "Maxiter": 50,
        # "func" : CEC2014benchmark.Shifted_Rotated_Expanded_Scaffer_F6,
    }
    bound = np.tile([[-100], [100]], params["vardim"])
    # bound = np.array([[2.6, 0.7, 17, 7.3, 7.8, 2.9, 5],[3.6, 0.8, 28, 8, 9.3, 3.9, 5.5]])
    # params["func"] = TempTestFunction.SpeedProblem

    # bound = np.array([[0.0625, 0.0625, 10, 10], [99 * 0.0625, 99 * 0.0625, 200, 200]])
    params["func"] = Func.powX

    # bound = np.array([[78,33,27,27,27],[102,45,45,45,45]])
    # params["func"] = TempTestFunction.Himmelblau_1

    params["bound"] = bound

    import matplotlib.pyplot as plt

    PIO = SFPIO(params=params)
    x,y= PIO.solve()

    for i in range(50):
        plt.plot(x[:i+1], y[:i+1], c='g', marker='.')
        plt.show()
        time.sleep(0.05)









