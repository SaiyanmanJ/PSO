# coding: utf-8
# python 3.6.12
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import time

#---------------------PSO算法--------------------#
class PSO:
    def __init__(self, n, m, w, a1, a2, vmin, vmax, kmax, data, startPoint): # 初始化粒子群体状态
        self.dim = n # 维度
        self.amount = m # 粒子数量
        self.w = w # 从粒子种群中选择个数
        '''
        self.c1 = c1 # 认知部分加速系数
        self.c2 = c2 # 社会部分加速系数
        '''
        self.a1 = a1 # 随机控制参数
        self.a2 = a2
        self.vmin = vmin # 最小速度
        self.vmax = vmax # 最大速度
        self.maxiterations = kmax # PSO最大迭代次数

        self.X = [[] for i in range(self.amount)] # 粒子 x 初始化
        self.V = [[] for i in range(self.amount)] # 粒子 v 初始化

        self.fit = [0 for i in range(self.amount)] # 适应值
        self.ie = [[] for i in range(self.amount)] # 个体找到的极值点的位置
        self.iefit = [1e10 for i in range(self.amount)] # 每个粒子找到的历史极值点
        self.ge = [] # 全局极值点的位置
        self.gefit = 1e10 # 全局极值
        self.gefits = [] # 记录每次迭代的全局极值  
        self.bestGeTimes = 0 # 最优解出现需要的迭代次数
        self.data = data
        self.distance = np.zeros((len(self.data), len(self.data)))
        self.population = range(0, self.dim)
        self.startPoint = startPoint

    # 计算任意两个城市之间的距离（经纬度）
    def caldistance(self): 
        PI = 3.141592
        RRR = 6378.388
        C1 = PI/180.0
        C2 = 5/3.0

        # 取任意经纬度坐标的整数部分和小数部分
        integerData = np.zeros((len(self.data),2))
        decimalData = np.zeros((len(self.data),2))
        
        m = len(self.data)
        i = 0
        while(i < m):
            j = 0
            while(j < 2):
                decimalData[i][j], integerData[i][j] = math.modf(self.data[i][j])
                j += 1
            i += 1
        
        # 计算两个城市之间的距离
        i = 0
        while(i < m):
            j = i + 1
            while(j < m):
    
                q1 = math.cos(C1 * (integerData[i][1] - integerData[j][1] + C2 * (decimalData[i][1] - decimalData[j][1])))
                q2 = math.cos(C1 * (integerData[i][0] - integerData[j][0] + C2 * (decimalData[i][0] - decimalData[j][0])))
                q3 = math.cos(C1 * (integerData[i][0] + integerData[j][0] + C2 * (decimalData[i][0] + decimalData[j][0])))
                
                self.distance[i][j] = round(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
                self.distance[j][i] = self.distance[i][j]
                
                j += 1
            i += 1
        
    # 初始化种群
    def init_swarm(self): 
        
        for i in range(self.amount):
            # 初始化X 起点城市默认为 1 号城市即 0
            self.X[i] =[0] + random.sample(self.population[1:],self.dim - 1) #表示从[0,dim]间随机生成dim个数，结果以列表返回
            
            for j in range(self.amount): 
                # 初始化V
                self.V[i].append(tuple(random.sample(self.population[1:],2)))

            self.ie[i] = copy.deepcopy(self.X[i]) # 个体极值点的位置设置为当前位置
            self.fit[i] = self.function(self.X[i]) # 适应度值
            self.iefit[i] = self.fit[i] # 个体极值
            if self.iefit[i] < self.gefit: # 更新全局极值
                self.gefit = self.iefit[i]
                self.ge = copy.deepcopy(self.ie[i])
    # 迭代
    def run(self):
        
        for iterTime in range(self.maxiterations): # 迭代 maxiterations 次
            for i in range(self.amount): 
                
                # 计算粒子适应度值
                self.fit[i] = self.function(self.X[i]) 

                # 更新个体极值
                if self.fit[i] < self.iefit[i]:
                    self.iefit[i] = self.fit[i]
                    self.ie[i] = copy.deepcopy(self.X[i])
                    # 找到全局极值
                    if self.iefit[i] < self.gefit:
                        self.gefit = self.iefit[i]
                        self.ge = copy.deepcopy(self.ie[i])
                        self.bestGeTimes = iterTime + 1

            for i in range(self.amount):
                # 更新粒子速度
                self.V[i] = self.rand(self.w, self.V[i]) + self.rand(random.uniform(0, self.a1),self.sub(self.ie[i], self.X[i])) + self.rand(random.uniform(0, self.a2), self.sub(self.ge, self.X[i]))
                # self.V[i] = self.rand(self.w, self.V[i]) + self.rand(1,self.sub(self.ie[i], self.X[i])) + self.rand(1, self.sub(self.ge, self.X[i]))
                
                # 更新粒子位置
                self.X[i] = self.add(self.X[i], self.V[i])
            
            self.gefits.append(self.gefit) # 记录当前迭代的全局极值
            #print(self.ge, end=" ")
            print(f"{iterTime} {self.gefit}\n")  # 输出最优值
        return self.gefits, self.ge, self.bestGeTimes, self.gefit

    # 以概率p，保留es的交换子，即保留前p个es的交换子
    def rand(self,probability, es):
        temp = []
        for i in range(round(probability * len(es))): # 下标不大于 probability * len(es)
            temp.append(es[i])
        return temp
    # 定义加法 按照交换子 交换 x 中对应位置
    def add(self, x, es): 
        for co in es:
            temp = x[co[0]]
            x[co[0]] = x[co[1]]
            x[co[1]] = temp
        return x

    # 求交换子  同一个城市 k 在不同序列中的不同下标 (i, j)
    def getCommutator(self, x1, x2):
        for i in range(len(x1)):
            for j in range(len(x2)):
                if i != j and x1[i] == x2[j]:
                    return (i, j)
    # 定义减法 求交换序列
    def sub(self, x1, x2):
        es = []
        t1 = x1; t2 = copy.deepcopy(x2)
        
        while(t1 != t2): # 不相同就一直求交换序列
            co = self.getCommutator(t1, t2)
            es.append(co)
            t2 = self.add(t2, [co]) # 更新t2
        return es

    # 目标函数 适应度值即为路径长度
    def function(self, s): # tsp 计算城市序列s的路线长度
        dis = 0
        m = len(s)
        for i in range(m):
            dis += self.distance[s[i % m]][s[(i + 1) % m]]
        return  dis

# 读入数据，每个城市的坐标,并计算每个城市之间的距离
dataPath = 'burma14.txt' # 最优解 3323
data = []
with open(dataPath, encoding='utf-8') as f:
    data = np.loadtxt(f)
    data = data[:, 1:3]
    # print(data)
    
#-----------------------主程序---------------------------------------------
begin = time.clock()

maxiterations = 10000
pso = PSO(14, 300, 0.7298, 1, 1, 0, 1, maxiterations, data, 1)
pso.caldistance()
pso.init_swarm()
fitness, ge, bestGeTimes, gefit = pso.run()
# fitness = np.array(fitness)

end = time.clock()
print(f"running time: {end - begin}")
# -------------------画图--------------------
# ------------------迭代次数与路径长度的变化图-------------------
plt.figure(1)
plt.title("Iterators and Road Length")
plt.xlabel("Iterators", size=14)
plt.ylabel("Road Length", size=14)
plt.scatter(bestGeTimes, gefit, color='r')
plt.text(bestGeTimes, gefit, str((bestGeTimes, gefit)))
# t = np.array([t for t in range(0, maxiterations)])
t = [i for i in range(1, maxiterations + 1)]
plt.plot(t, fitness, color='b', linewidth=3)
plt.show()
# ------------------路线图--------------------------------------
# 红色点为起始点

x = []
y = []
for i in ge:
    x.append(data[i][0])
    y.append(data[i][1])
plt.figure(1)
plt.title("City Path")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
# plt.plot(x, y, color='r')

plt.scatter(x, y, color='b') # 点
plt.scatter(x[0],y[0], color='r') # 起始点
for i in range(len(ge)):
    plt.text(x[i], y[i], ge[i])
n = len(x)
for i in range(n):
    # plt.arrow(x[i], y[i],x[(i + 1)%n] - x[i],y[(i + 1)%n] - y[i], color='b')
    plt.annotate("",xy=(x[(i + 1)%n], y[(i + 1)%n]),xytext=(x[i], y[i]), arrowprops=dict(arrowstyle="->")) # 加箭头
plt.show()
# %%
