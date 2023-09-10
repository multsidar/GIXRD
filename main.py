import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import cmath


def frenel(a):
    a = np.radians(a)
    return abs(2 * complex(a)/(a+cmath.sqrt(complex(a ** 2 - 2 * delta,-2*beta))))**2

def tau(a):
    return wavelen * np.sqrt(2) / (4 * np.pi) \
              * (np.sqrt((np.radians(a) ** 2 - 2 * delta) ** 2 + 4 * beta ** 2)
                 - (np.radians(a) ** 2 - 2 * delta)) ** -0.5


def intensity_tf1_big (X,c,t):
    a, theta_2 = X
    return c * tau(a) / (lin_koef * np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1) / np.sin(np.radians(a)) * \
           (np.sin(np.radians(theta_2 - a)) / (np.sin(np.radians(theta_2 - a)) + np.sin(np.radians(a)))) * \
           (1 - np.exp(- t * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))


fp = lambda a,length,width: min(length / width * np.sin(np.radians(a)), 1)

#________Const_______________#
delta = 3.51 * 10 ** -5
beta = 2.67 * 10 ** -6
lin_koef = 2184 * 10 ** -6


wavelen = .154
width = 0.6
length = 40
beta2=np.sqrt(complex(-1))*beta

#___________Data_____________#
sheet = pd.read_excel("Ru_31nm_0.6mm_45.xlsx",usecols=[0,1,2,3,4,5,6,7])
df = pd.DataFrame(sheet.values)
#print(df)


z1 = list(map(float,df[6].values))
z1 = [x for x in z1 if str(x) != 'nan']

angle = list(map(float,df[7].values))
angle = [x for x in angle if str(x) != 'nan']

theta_2 = list(map(float,df[1].values))
theta_2 = [x for x in theta_2 if str(x) != 'nan']

#__________Plot___________#
value1 = np.empty([len(angle)], float)

X = (angle,theta_2)
popt1, pcov1 = curve_fit(intensity_tf1_big, X, z1)
c1 = popt1[0]
t1 = popt1[1]
print(t1)


for i, alph in enumerate(angle):
    frn=1
    frn=(abs(frenel(alph)))
    X1=[alph,theta_2[i]]
    value1[i] = intensity_tf1_big(X1, c1, t1) * frn * fp(alph,length,width)


plt.plot(angle,z1,'o')
plt.plot(angle,value1,'-')
plt.show()







