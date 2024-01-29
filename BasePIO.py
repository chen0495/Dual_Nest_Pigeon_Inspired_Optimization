from .Population import Pops
import numpy as np
import time

class BasePIO():
    def __init__(self,params,**kwargs):
        '''
        :param kwargs: dict
            such as:
                {
                "popSize" : 40
                "vardim": 50
                "bound" : [[l_1,l_2,...,l_n],[u_1,u_2,...,u_n]] # and n=vardim
                "Maxiter" : 1000
                "R" : map factor,default=0.2
                "init_type" : 0 for oringin、1 for latin
                "personal_type": True with Personal_Best updated
                "func" : function of calculating fitness
                "M" : matrix of input
                "S" : shuffle of input
                }
        '''
        #print(params,kwargs)
        self.popSize, self.vardim, self.bound = 40, 50, np.tile([[-100], [100]], 50)
        self.func, self.M, self.S = None, np.eye(self.vardim),np.zeros(self.vardim)
        self.init_type,self.personal_type = 0,False
        self.Maxiter = 1000


        self.__set_keyword_arguments(params)
        self.__set_keyword_arguments(kwargs)

        self.Nc1 = int(0.8*self.Maxiter)
        self.Nc2 = self.Maxiter-self.Nc1
        self.Pops = Pops(params)
        self.half = self.popSize
        self.R = 0.2

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def move_part_one(self,epoch):
        self.Pops.v = self.Pops.v*np.exp(-self.R * epoch) + np.random.random((self.popSize,self.vardim))*\
                      (self.Pops.global_best_position-self.Pops.pop)

        self.Pops.pop = self.Pops.v + self.Pops.pop

    def move_part_two(self,epoch):
        better_pop_center = self.Pops.calculate_center(self.half)

        self.Pops.pop = self.Pops.pop+np.random.random((self.popSize,self.vardim))*(better_pop_center-self.Pops.pop)

    def reduce_better_populations_size(self):
        self.half = int(self.half/2)+1

    def solve(self):
        x = np.zeros((self.popSize,self.Maxiter))
        y = np.zeros((self.popSize,self.Maxiter))
        print("***{}***".format(self.__class__.__name__))
        st = time.time()
        for t in range(self.Nc1):           #指南针算子
            self.move_part_one(t)            #更新种群位置
            self.Pops.updated()             #更新种群状态
            x[:, t] = self.Pops.pop[:, 0]
            y[:,t] = self.Pops.pop[:, 1]
        print("Part.{},  fitness = {}".format(t,self.Pops.global_best_fitness))
        #print("pop[:10] = ", self.Pops.pop[:10],self.Pops.fitness[:10])

        for t in range(self.Nc2):
            self.reduce_better_populations_size()   #优势种群减半
            self.move_part_two(t)
            self.Pops.updated()
            x[:, t + self.Nc1] = self.Pops.pop[:, 0]
            y[:, t+self.Nc1] = self.Pops.pop[:, 1]
        #print("Part.{},  fitness = {}, position = {}".format(t+self.Nc1,self.Pops.global_best_fitness,self.Pops.global_best_position))

        et = time.time()
        print("run_time={}s, best_fitness = {}".format(et-st,self.Pops.global_best_fitness))
        return x, y
        #return self.Pops.global_best_fitness,self.Pops.global_best_position



if __name__ == "__main__":
    import TempTestFunction as Func

    params = {
        "popSize": 20,
        "vardim": 2,
        "Maxiter": 100,
        # "func" : CEC2014benchmark.Shifted_Rotated_Expanded_Scaffer_F6,
    }
    bound = np.tile([[-100], [100]], params["vardim"])
    # bound = np.array([[2.6, 0.7, 17, 7.3, 7.8, 2.9, 5],[3.6, 0.8, 28, 8, 9.3, 3.9, 5.5]])
    # params["func"] = TempTestFunction.SpeedProblem

    # bound = np.array([[0.0625, 0.0625, 10, 10], [99 * 0.0625, 99 * 0.0625, 200, 200]])
    params["func"] = Func.powX_90

    # bound = np.array([[78,33,27,27,27],[102,45,45,45,45]])
    # params["func"] = TempTestFunction.Himmelblau_1

    params["bound"] = bound

    record = []
    for i in range(10):
        PIO = BasePIO(params=params)
        PIO.solve()
        print(PIO.Pops.global_best_position)
        record.append(PIO.Pops.global_best_fitness)

    print(np.mean(record))






