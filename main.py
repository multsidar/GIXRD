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


def intensity_tf1_big (a,c,t ):
    return c * tau(a) / np.sin(np.radians(a))* \
           (np.sin(np.radians(theta_2 - a)) / (np.sin(np.radians(theta_2 - a)) + np.sin(np.radians(a)))) * \
           (1-np.exp(- t*2 * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))* \
           (np.exp(- t*2 * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))


def intensity_tf1_big1 (a,c,t ):
    return c /(- t * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1))* \
           (np.sin(np.radians(theta_2 - a)) / (np.sin(np.radians(theta_2 - a)) + np.sin(np.radians(a)))) * \
           (1-np.exp(- t * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))

def intensity_tf1_small (a,c,t):
    return c * tau(a) / np.sin(np.radians(a))* \
           (np.sin(np.radians(theta_2 - a)) / (np.sin(np.radians(theta_2 - a)) + np.sin(np.radians(a)))) * \
           (1-np.exp(- t*2 * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))* \
           (np.exp(- t * 2 * lin_koef * (np.sin(np.radians(a)) ** -1 + np.sin(np.radians(theta_2 - a)) ** -1)))




def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


fp = lambda a,length,width: min(length / width * np.sin(np.radians(a)), 1)






delta =3.9806 * 10 ** -5
beta = 3.2277 * 10 ** -6
c = 15
t = 89
lin_koef = 2634 * 10 ** - 7

a = np.arange(0.3, 36, .01)
wavelen = .154
width = 0.6
length = 20
theta_2 = 38.5
beta2=np.sqrt(complex(-1))*beta

sheet = pd.read_excel("Ta.xlsx",usecols=[0,1,2,3])
df = pd.DataFrame(sheet.values)


angle = np.arange(0.3, 3.6, .1)


# облученый подложка
z1 = list(map(float,df[3].values))
z1 = [x for x in z1 if str(x) != 'nan']

# облученый пленка
z2 = list(map(float,df[2].values))
z2 = [x for x in z2 if str(x) != 'nan']

# Не облученный
z3 = list(map(float,df[3].values))
z3 = [x for x in z3 if str(x) != 'nan']
# Графики ##############################################################################################################
value1 = np.empty([len(a)], float)
value2 = np.empty([len(a)], float)
value3 = np.empty([len(a)], float)
t1=89
t2=89
t3=42
c1=0.6
c2=1.5
c3=1

popt1, pcov1 = curve_fit(intensity_tf1_big, angle, z1)
c1 = popt1[0]*1.
t1=popt1[1]


popt2, pcov2 = curve_fit(intensity_tf1_big, angle, z1)
c2 = popt2[0]*1.0
t2=popt2[1]

popt3, pcov3 = curve_fit(intensity_tf1_big, angle, z1)
c3 = popt3[0]*1.
t3=popt3[1]

popt4, pcov4 = curve_fit(intensity_tf1_small, angle, z1)
c4 = popt4[0]
t4=popt4[1]


popt5, pcov5 = curve_fit(intensity_tf1_small, angle, z1)
c5 = popt5[0]*0.5
t5=popt5[1]

popt6, pcov6 = curve_fit(intensity_tf1_small, angle, z1)
c6 = popt6[0]
t6=popt6[1]


for i, j in enumerate(a):
    frn=1
    frn=(abs(frenel(j)))
    if(fp(j,20,0.6)==1):
        value2[i] = intensity_tf1_big(j, c2, 40)*frn
        value3[i] = intensity_tf1_big(j, c3, 60)*frn
    else:
        value2[i] = intensity_tf1_big(j, c2, 40)*frn
        value3[i] = intensity_tf1_big(j, c3, 60)*frn
    if(fp(j,20,0.6)==1):
        value1[i] = intensity_tf1_big(j, c1, 20)*frn
    else:
        value1[i] = intensity_tf1_big(j, c1, 20)*frn


print(int(t1))
print(int(t2))
print(int(t3))
print(t4)
print(t5)
print(t6)
plt.subplots(1)
plt.plot(a, smooth(value1,10), label='theory 20 nm')
plt.plot(a, smooth(value2,10), label='approximation 40 nm')
plt.plot(a, smooth(value3,10), label='theory 60 nm')

plt.plot(angle,z1,'o',label='experiment')
#plt.plot(angle,z2,'o')
#plt.plot(angle,z3,'o')
ax = plt.gca()
ax.set_xlim([0.3, 3.6])

plt.ylabel('Интенсивность в отн. ед.')
plt.xlabel('Угол падения в град.')
plt.legend()
plt.grid(True)
plt.show()





