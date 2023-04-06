import math
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd


# поправка к углу тета
d2t = lambda a: a - 1/math.sqrt(2) * math.sqrt((a**2-2*delta) +
                                               math.sqrt((a**2-2*delta) ** 2 + 4 * beta ** 2))


# глубина проникновения рентгена
tau = lambda a: wavelen * math.sqrt(2) / (4 * math.pi) \
              * (math.sqrt((math.radians(a) ** 2 - 2 * delta) ** 2 + 4 * beta ** 2)
                 - (math.radians(a) ** 2 - 2 * delta)) ** -0.5


# эфф следа пучка
fp = lambda a: min(l/w * math.sin(math.radians(a)), 1)


# от угла падения учитывая глубину образца
intensity_a = lambda a: k * frenel(a) * fp(a) * tau(a) * (1 - math.exp(-h/tau(a))) / (2 * math.sin(math.radians(a)))


# Френелевский коэф прохождения
frenel = lambda a: 2 * math.sin(math.radians(a)) / (math.sin(math.radians(a))
                                                    * math.sqrt(math.sin(math.radians(a)) - 2 * delta))

# без учета толщины и френеля
intensity = lambda a: fp(a) * tau(a) / math.sin(math.radians(a) * 2)


#lhotka
l_intensity_tf = lambda a: c * (math.sin(math.radians(theta + a) / (math.sin(math.radians(theta - a) + math.sin(math.radians(theta + a)))))) * \
                           (math.exp(-m_TiN * t * (math.sin(math.radians(theta + a)) ** -1 + math.sin(math.radians(theta - a)) ** -1)))


l_intensity_sub = lambda a:  c * (math.sin(math.radians(theta - a) / (math.sin(math.radians(theta - a) + math.sin(math.radians(theta + a)))))) * \
                           (1-math.exp(-m_TiN * t * (math.sin(math.radians(theta + a)) ** -1 + math.sin(math.radians(theta - a)) ** -1)))

# Входные данные #########################################################################################################
delta = 1.28 * 10 ** -5
beta = 1.11 * 10 ** -6
a = np.arange(0.4, 35, .1)
wavelen = .154
h = 2 * 10 ** 3
k = 1
w = 0.15
l = 12.7
theta =42
c = 134
t = 0.31
m_TiN = 3.5 * 10 ** -2
# data##################################################################################################################
'''

argon_df = pd.DataFrame(pd.read_excel('EP_1003_1000.xlsx', sheet_name='Argon', usecols='C:I', header=39, nrows=8).values)
alpha = list(map(float,argon_df[0].values))
net_area_2_10_11 = list(map(float,argon_df[5].values))


argon_df2 = pd.DataFrame(pd.read_excel('EP_1003_1000.xlsx', sheet_name='Krypton', usecols='C:I', header=42, nrows=8).values)
net_area_2_10_12 = list(map(float,argon_df2[5].values))


argon_df3 = pd.DataFrame(pd.read_excel('EP_1003_1000.xlsx', sheet_name='Xenon', usecols='C:I', header=44, nrows=8).values)
net_area_6_10_10 = list(map(float,argon_df3[5].values))


#argon_df4 = pd.DataFrame(pd.read_excel('EP_1003_1000.xlsx',sheet_name=name,usecols='C:I',header=44,nrows=8).values)
#net_area_6_10_11 = list(map(float,argon_df4[5].values))

'''
# Графики ##############################################################################################################
value = np.empty([len(a)], float)
value1 = np.empty([len(a)], float)
a1 = np.empty([len(a)], float)
for i, t in enumerate(a):
    value[i] = l_intensity_tf(t)


plt.subplots(1)
plt.plot(a, value,label='theory')
#plt.plot(alpha,net_area_2_10_11,'o',label='Argon')
#plt.plot(alpha,net_area_2_10_12,'o',label='Krypton')
#plt.plot(alpha,net_area_6_10_10,'o',label='Xenon')
#plt.plot(alpha,net_area_6_10_11,'o',label='6_10_11')
plt.ylabel('Интенсивность в отн. ед.')
plt.xlabel('Угол падения в град.')
plt.legend()
plt.show()
#my_path = os.path.abspath('D:\GIXRD')
#my_file = 'GIXRD_Argon.jpeg'
#plt.savefig(os.path.join(my_path, my_file))