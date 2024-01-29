import numpy as np
import copy

class Pops():
    def __init__(self, params,**kwargs):
        '''
        :param kwargs: dict
            such as:
                {
                "popSize" : 40
                "vardim": 50
                "bound" : [[l_1,l_2,...,l_n],[u_1,u_2,...,u_n]] # and n=vardim
                "init_type" : 0 for oringin、1 for latin
                "personal_type": True with Personal_Best updated
                "func" : function of calculating fitness
                "M" : matrix of input
                "S" : shuffle of input
                "history" : True with historical global_best_fitness recorded
                "omiga" : 1e-8
                }
        '''
        self.popSize, self.vardim, self.bound = 40, 50, np.tile([[-100],[100]],50)
        self.func,self.M,self.S = None,np.eye(self.vardim),np.zeros(self.vardim)
        self.init_type,self.personal_type = 0,False
        self.global_best_fitness_history,self.history,self.population_history, =[],True,[]
        self.exploration_history, self.exploitation_history, self.d_history = [],[],[]
        self.omiga = 1e-10
        self.failue_count = 0
        self.absorb_p = 0.9
        self.update_pbest = True

        self.__set_keyword_arguments(params)
        self.__set_keyword_arguments(kwargs)
        self.bound = np.array(self.bound) # to narray
        self.solution = {
            0 : self.oringin,
            1 : self.latin
        }
        self.fitness, self.global_best_fitness, self.personal_best_fitness = None,None,None
        self.global_best_position, self.personal_best_position = None,None

        self.a_min = np.tile(self.bound[0], (self.popSize, 1))
        self.a_max = np.tile(self.bound[1], (self.popSize, 1))
        self.a_max_min = self.a_max-self.a_min
        #print(kwargs)

        self.pop = self.solution[self.init_type]()  # 种群初始化
        self.v = self.bound[0]+np.random.random((self.popSize,self.vardim))*(self.bound[1]-self.bound[0]) #速度初始化
        self.update_firstly()


        #self.updated() #更新or排序：fitness、global_best_fitness、personal_best_fitness、global_best_position、personal_best_position、



    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def oringin(self):
        pop = np.random.uniform(self.bound[0],self.bound[1],size=(self.popSize,self.vardim))#np.random.random((self.popSize,self.vardim))*(self.bound[1]-self.bound[0])+self.bound[0]
        #print("o=",pop)
        return pop

    def latin(self):
        step = np.linspace(self.bound[0],self.bound[1],self.popSize+1)
        pop = np.random.uniform(step[0:self.popSize],step[1:self.popSize+1],(self.popSize,self.vardim))
        #print("l=",pop)
        for i in range(self.vardim): #打乱每一维
            np.random.shuffle(pop[:, i])

        return pop

    def border_check(self,*X):
        pop = self.oringin()
        if X==():
            X = self.pop
        idx = (X<self.a_min)
        X[idx] = pop[idx]
        idx = (self.pop>self.a_max)
        X[idx] = pop[idx]
        self.pop = X
#         self.pop = np.clip(self.pop,a_min=self.bound[0],a_max=self.bound[1])


    
    def border_check_model2(self):
        self.pop = np.clip(self.pop, a_min=self.a_min, a_max=self.a_max)

    def border_check_damping(self):
        idx = (self.pop < self.a_min)
        temp = (self.a_min[idx]-self.pop[idx])
        temp = np.minimum(temp,self.a_max_min[idx])
        self.pop[idx] = temp*self.absorb_p+self.a_min[idx]

        idx = (self.pop > self.a_max)
        temp = (self.pop[idx]-self.a_max[idx])
        temp = np.minimum(temp, self.a_max_min[idx])
        self.pop[idx] = -temp*self.absorb_p + self.a_max[idx]

    def border_check_ones(self,x):
        if np.any(x<self.bound[0]) or np.any(x>self.bound[1]):
                x = np.random.uniform(self.bound[0],self.bound[1],size=self.vardim)
        return x

    def update_firstly(self):
        self.border_check()  # 边界检查
        self.fitness = np.zeros(self.popSize)
        self.fitness = self.func(self.pop)

        index = self.__sort_fitness()
        

        self.personal_best_position = copy.deepcopy(self.pop)
        self.personal_best_fitness = copy.deepcopy(self.fitness)
        self.global_best_position = copy.deepcopy(self.pop[0])
        self.global_best_fitness = copy.deepcopy(self.fitness[0])

    def calculate_fitness_ones(self,x):
        return self.func([x])

    def calculate_fitness_borderAlt(self):
        self.border_check_damping() #边界检查
        self.fitness = np.zeros(self.popSize)
        self.fitness = self.func(self.pop)

        idx = self.personal_best_fitness>self.fitness
        self.personal_best_fitness[idx] = copy.deepcopy(self.fitness[idx])
        self.personal_best_position[idx] = copy.deepcopy(self.pop[idx])

        idx = np.argmin(self.personal_best_fitness)
        self.global_best_fitness = copy.deepcopy(self.personal_best_fitness[idx])
        self.global_best_position = copy.deepcopy(self.personal_best_position[idx])

        return self.fitness

    def calculate_fitness(self):
        self.border_check() #边界检查
        self.fitness = np.zeros(self.popSize)
        self.fitness = self.func(self.pop)
        #print(self.fitness,self.personal_best_fitness)
        idx = np.where(self.personal_best_fitness > self.fitness)
        self.personal_best_fitness[idx] = copy.deepcopy(self.fitness[idx])
        self.personal_best_position[idx] = copy.deepcopy(self.pop[idx])

        idx = np.argmin(self.personal_best_fitness)
        self.global_best_fitness = copy.deepcopy(self.personal_best_fitness[idx])
        self.global_best_position = copy.deepcopy(self.personal_best_position[idx])


        return self.fitness

    def calculate_fitness_no_check(self):
        self.fitness = np.zeros(self.popSize)
        flag = False
        self.fitness = self.func(self.pop)
        idx = self.personal_best_fitness > self.fitness
        self.personal_best_fitness[idx] = copy.deepcopy(self.fitness[idx])
        self.personal_best_position[idx] = copy.deepcopy(self.pop[idx])

        idx = np.argmin(self.personal_best_fitness)
        self.global_best_fitness = copy.deepcopy(self.personal_best_fitness[idx])
        self.global_best_position = copy.deepcopy(self.personal_best_position[idx])
        return self.fitness

    def __sort_fitness(self):
        index = np.argsort(self.fitness)
        self.fitness = self.fitness[index]
        self.pop = self.pop[index]
        self.v = self.v[index]
        return index

    def sort_pop_by_fitness(self):
        index = self.__sort_fitness()
        if self.update_pbest:
            self.personal_best_position = copy.deepcopy(self.personal_best_position[index])
            self.personal_best_fitness = copy.deepcopy(self.personal_best_fitness[index])


    def calculate_center(self,n=None):
        '''
        Please Note that the populations must be sorted by fitness
        :param n: nums of better populations
        '''
        if n==None:
            n=int(self.popSize/2)+1
        S = 0
        F = 0
        minf = np.min(self.fitness)
        for i in range(n):
            temp_fit = 1/(self.fitness[i]-minf+self.omiga)
            S = S+self.pop[i]*temp_fit
            F = F+temp_fit
        return S/(F*n)

    def calculate_center_2(self, n=None):
        '''
        Please Note that the populations must be sorted by fitness
        :param n: nums of better populations
        '''
        if n == None:
            n = int(self.popSize / 2) + 1
        index = np.argsort(self.personal_best_fitness)
        sorted_personal_best_fitness = copy.deepcopy(self.personal_best_fitness[index])
        sorted_personal_best_position = copy.deepcopy(self.personal_best_position[index])
        S = 0
        F = 0
        for i in range(n):
            temp_fit = 1 / (sorted_personal_best_fitness[i] + self.omiga)
            S = S + sorted_personal_best_position[i] * temp_fit
            F = F + temp_fit
        return S / F


        return np.mean(self.pop[:n],0)

    def pop_ee(self,flag=True):
        if flag==False:
            return
        self.global_best_fitness_history.append(self.global_best_fitness)
        self.population_history.append(((self.pop.mean(0) - self.pop) ** 2).sum() / self.vardim)

        self.d_history.append( np.sum( np.abs(np.median(self.pop,0)-self.pop) ) / ( self.popSize*self.vardim ) )
        d_history_max = np.max(self.d_history)
        self.exploration_history.append(100*self.d_history[-1]/d_history_max)
        self.exploitation_history.append(100*np.abs(self.d_history[-1]-d_history_max)/d_history_max)

    def updated(self):
        self.calculate_fitness()
        self.sort_pop_by_fitness()
        self.pop_ee()



    def updated_with_rool_back(self,X,Xfit,v,flag=True):
        self.calculate_fitness()
        idx = np.where(Xfit<self.fitness)
        self.pop[idx] = copy.deepcopy(X[idx])
        self.fitness[idx] = copy.deepcopy(Xfit[idx])
        self.v[idx] = copy.deepcopy(v[idx])
        self.sort_pop_by_fitness()
        self.pop_ee(flag)

    
    def updated_2(self):
        self.calculate_fitness_borderAlt()
        self.sort_pop_by_fitness()
        self.pop_ee()


    # def updated_2(self):
    #     self.calculate_fitness_borderAlt()
    #     #self.sort_pop_by_fitness()
    #     if self.history:
    #         self.global_best_fitness_history.append(self.global_best_fitness)
    #         self.population_history.append(((self.pop.mean(0)-self.pop)**2).sum()/self.vardim)