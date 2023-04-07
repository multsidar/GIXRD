import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd


# поправка к углу тета
d2t = lambda a: a - 1/np.sqrt(2) * np.sqrt((a**2-2*delta) +
                                               np.sqrt((a**2-2*delta) ** 2 + 4 * beta ** 2))


# глубина проникновения рентгена
tau = lambda a: wavelen * np.sqrt(2) / (4 * np.pi) \
              * (np.sqrt((np.radians(a) ** 2 - 2 * delta) ** 2 + 4 * beta ** 2)
                 - (np.radians(a) ** 2 - 2 * delta)) ** -0.5


# эфф следа пучка
fp = lambda a: min(length / width * np.sin(np.radians(a)), 1)


# от угла падения учитывая глубину образца
intensity_a = lambda a: k * frenel(a) * fp(a) * tau(a) * (1 - np.exp(-h/tau(a))) / (2 * np.sin(np.radians(a)))


# Френелевский коэф прохождения
frenel = lambda a: 2 * np.sin(np.radians(a)) / (np.sin(np.radians(a))
                                                    * np.sqrt(np.sin(np.radians(a)) - 2 * delta))

# без учета толщины и френеля
intensity = lambda a: fp(a) * tau(a) / np.sin(np.radians(a) * 2)


#lhotka
l_intensity_tf = lambda a: c * (np.sin(np.radians(theta - a) / (np.sin(np.radians(theta - a) + np.sin(np.radians(a)))))) * \
                           (1-np.exp(-m_TiN * t * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta - a)) ** -1)))


l_intensity_sub = lambda a:  c * (np.sin(np.radians(theta - a) / (np.sin(np.radians(theta - a) + np.sin(np.radians(a)))))) * \
                           (np.exp(-m_TiN * t * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta - a)) ** -1)))

# Входные данные #########################################################################################################
delta = 1.28 * 10 ** -5
beta = 1.11 * 10 ** -6
a = np.arange(2, 30, .1)
wavelen = .154
h = 2 * 10 ** 3
k = 1
width = 0.15
length = 12.7
theta =33
c = 97
t = 2.03
m_TiN = 5.22 * 10 ** -2

# Графики ##############################################################################################################
value = np.empty([len(a)], float)
value1 = np.empty([len(a)], float)
a1 = np.empty([len(a)], float)
for i, j in enumerate(a):
    value[i] = l_intensity_tf(j)


plt.subplots(1)
plt.plot(a, value,label='theory')
plt.ylabel('Интенсивность в отн. ед.')
plt.xlabel('Угол падения в град.')
plt.legend()
my_path = os.path.abspath('D:\GIXRD_graph\lhotka')
my_file = 'GIXRD_Lhotka_sub_t=' + str(t) +'m_TiN'+ str(m_TiN) +'.png'
plt.savefig(os.path.join(my_path, my_file))
plt.show()
