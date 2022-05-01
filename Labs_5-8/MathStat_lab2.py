import scipy.stats as sps
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import statistics as stats
import math
from matplotlib.patches import Ellipse

def SquareCoefCor(x, y):
    n1 = 0
    n2 = 0
    n3 = 0
    n4 = 0
    medx = stats.median(x)
    medy = stats.median(y)
    for i in range(len(x)):
        if x[i] > medx and y[i] > medy: n1 += 1
        elif x[i] < medx and y[i] > medy: n2 += 1
        elif x[i] < medx and y[i] < medy: n3 += 1
        elif x[i] > medx and y[i] < medy: n4 += 1

    return ((n1 + n3) - (n2 + n4)) / len(x)

def SquareMean(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i] * x[i]
    return ans / len(x)

def Step1():

    mean = [0.0, 0.0]  # mean of x and y

    ro = np.array([0, 0.5, 0.9])
    sample_size = np.array([20, 60, 100])
    size = 1000

    r = np.array([], dtype = float)
    rs = np.array([], dtype = float)
    rq = np.array([], dtype = float)
    
    for i in range(len(sample_size)):

        print("n = ", sample_size[i], end = '\n')

        for j in range(len(ro)):
            cov = [[1.0 ,ro[j]], [ro[j], 1.0]]
            print("ro = ", ro[j], end = '\n')

            for k in range(size):

                rv = sps.multivariate_normal.rvs(mean, cov, sample_size[i])
                x = rv[:, 0]
                y = rv[:, 1]
                #коэфы корреляции Пирсона, Спирмена и квадрантный коэф
                #корреляции.

                a = sps.pearsonr(x, y)
                r = np.append(r, a[0])

                a = sps.spearmanr(x,y).correlation
                rs = np.append(rs, a)

                rq = np.append(rq, SquareCoefCor(x,y))

            #среднее значение, среднее значение квадрата и дисперсия
            print("r", "rs", "rq", sep = '\t', end = '\n')

            print("Ez", round(stats.mean(r), 3), round(stats.mean(rs), 3),
            round(stats.mean(rq), 3), sep = ' ', end = '\n')
            print("Ez2", round(SquareMean(r), 3), round(SquareMean(rs), 3),
            round(SquareMean(rq), 3), sep = ' ', end = '\n')
            print("Dz", round(stats.variance(r), 3), round(stats.variance(rs),
            3), round(stats.variance(rq), 3), sep = ' ', end = '\n')
            print(end='\n\n')

        r = np.array([], dtype = float)
        rs = np.array([], dtype = float)
        rq = np.array([], dtype = float)

    #смесь распределений
    mean = [0, 0]
    cov1 = [[1.0 ,0.9], [0.9, 1.0]]
    cov2 = [[10.0, -0.9], [-0.9, 10.0]]

    for i in range(len(sample_size)):
         print("n = ", sample_size[i], end = '\n')
         for k in range(size):
             rv = np.concatenate((0.9 * sps.multivariate_normal.rvs(mean, cov1, sample_size[i]), 0.1 * sps.multivariate_normal.rvs(mean, cov2, sample_size[i])))
             x = rv[:, 0]
             y = rv[:, 1]
             #коэфы корреляции Пирсона, Спирмена и квадрантный коэф
             #корреляции.

             a = sps.pearsonr(x, y)
             r = np.append(r, a[0])

             a = sps.spearmanr(x,y).correlation
             rs = np.append(rs, a)

             rq = np.append(rq, SquareCoefCor(x,y))
         #среднее значение, среднее значение квадрата и дисперсия

         print("r", "rs", "rq", sep = '\t', end = '\n')
         print("Ez", round(stats.mean(r), 3), round(stats.mean(rs), 3), round(stats.mean(rq), 3), sep = ' ', end = '\n')
         print("Ez2", round(SquareMean(r), 3), round(SquareMean(rs), 3), round(SquareMean(rq), 3), sep = ' ', end = '\n')
         print("Dz", round(stats.variance(r), 3), round(stats.variance(rs), 3), round(stats.variance(rq), 3), sep = ' ', end = '\n')
         print(end='\n\n')

         r = np.array([], dtype = float)
         rs = np.array([], dtype = float)
         rq = np.array([], dtype = float)

def sgn(z):
    if z > 0: return 1
    elif z == 0: return 0
    else: return -1 #z<0

def Step2():
    n = 20
    x = np.linspace(-1.8, 2.0, num = n)
    eps = np.random.normal(loc = 0, scale = 1, size = n)
    y = 2 + 2 * x + eps

    #в случае возмущений:
    y[0] = y[0] + 10
    y[n - 1] = y[n - 1] - 10

    #МНК
    xy = x * y
    a = (xy.mean() - x.mean() * y.mean()) / (SquareMean(x) - x.mean() * x.mean())
    b = y.mean() - x.mean() * a
    print('МНК:', 'a = ', a, 'b = ', b)

    #Метод наименьших модулей
    rq = 0
    for i in range(n):
        rq += (sgn(x[i] - stats.median(x)) * sgn(y[i] - stats.median(y)))
    rq = rq / n

    zx = stats.quantiles(x, n=4)
    zy = stats.quantiles(y, n=4)

    kqn = 1.491 #const

    qy = (zy[2] - zy[0]) / kqn
    qx = (zx[2] - zx[0]) / kqn

    b1r = rq * qy / qx
    b2r = stats.median(y) - b1r * stats.median(x)
    print('МНМ:', 'a = ', b1r, 'b = ', b2r)

    #График
    plt.scatter(x, y, label = 'Выборка')
    plt.plot([-1.8, 2], [2 * (-1.8) + 2 + eps[0], 2 * 2 + 2 + eps[n - 1]], label = 'Модель', c='r')
    plt.plot([-1.8, 2], [a * (-1.8) + b, a * 2 + b], label = 'МНК', c = 'k')
    plt.plot([-1.8, 2], [b1r * (-1.8) + b2r, b1r * 2 + b2r], label = 'МНМ', c = 'b')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()

def Step3():

    #Нормальное распределение
    print('normal')
    n = 100
    k = 6
    x = np.random.normal(loc = 0, scale = 1, size = n)
    q = 11.07

    ai = np.array([-np.inf, -1.01, -0.37, 0.27, 0.92, 1.56, np.inf], dtype = float)
    ni = np.array([0,0,0,0,0,0], dtype = float)
    pi = np.array([], dtype = float)

    #Метод макс правдоподобия
    mu, sigma = sps.norm.fit(x)

    for i in range(n):
        if x[i] <= ai[1]: ni[0] = ni[0] + 1
        elif x[i] <= ai[2]: ni[1] = ni[1] + 1
        elif x[i] <= ai[3]: ni[2] = ni[2] + 1
        elif x[i] <= ai[4]: ni[3] = ni[3] + 1
        elif x[i] <= ai[5]: ni[4] = ni[4] + 1
        else: ni[5] = ni[5] + 1

    for i in range(6):
        pi = np.append(pi, [sps.norm.cdf([ai[i + 1]], 0, 1) - sps.norm.cdf([ai[i]], 0, 1)])

    ans = np.array([])
    print('ni:', *ni, np.sum(ni))
    print('pi: ', *pi, np.sum(pi))
    for i in range(6): pi[i] = pi[i] * n
    print('npi: ', *pi, np.sum(pi))
    for i in range(6): ni[i] = ni[i] - pi[i]
    print('ni - npi: ', *ni, np.sum(ni))
    for i in range(6): ans = np.append(ans, [ni[i]*ni[i] / pi[i]])
    print('ans: ', *ans, np.sum(ans))


    #Лаплас
    print()
    print('laplace')
    n = 20
    a = 0.05
    x = np.random.laplace(0, sigma/np.sqrt(2), size = n)
    k = 5
    q = 9.49

    ai = np.array([-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], dtype = float)
    ni = np.array([0,0,0,0,0,0], dtype = float)
    pi = np.array([], dtype = float)

    #Хи-квадрат
    for i in range(n):
        if x[i] <= ai[1]: ni[0] = ni[0] + 1
        elif x[i] <= ai[2]: ni[1] = ni[1] + 1
        elif x[i] <= ai[3]: ni[2] = ni[2] + 1
        elif x[i] <= ai[4]: ni[3] = ni[3] + 1
        elif x[i] <= ai[5]: ni[4] = ni[4] + 1
        else: ni[5] = ni[5] + 1

    for i in range(5):
        pi = np.append(pi, [sps.norm.cdf([ai[i + 1]], 0, 1) - sps.norm.cdf([ai[i]], 0, 1)])

    ans = np.array([], dtype = float)
    print('ni: ', *ni, np.sum(ni))
    print('pi: ',*pi, np.sum(pi))
    for i in range(5): pi[i] = pi[i] * n
    print('npi: ',*pi, np.sum(pi))
    for i in range(5): ni[i] = ni[i] - pi[i]
    print('ni - npi: ',*ni, np.sum(ni))
    for i in range(5): ans = np.append(ans, [ni[i]*ni[i] / pi[i]])
    print('ans: ',*ans, np.sum(ans))

    #Равномерное
    print()
    print('Uniform')
    n = 20
    a = 0.05
    sqrt3 = np.sqrt(3)
    x = np.random.uniform(-sqrt3, sqrt3, size = n)
    k = 5
    q = 9.49

    ai = np.array([-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf], dtype = float)
    ni = np.array([0,0,0,0,0,0], dtype = float)
    pi = np.array([], dtype = float)

    #Хи-квадрат
    for i in range(n):
        if x[i] <= ai[1]: ni[0] = ni[0] + 1
        elif x[i] <= ai[2]: ni[1] = ni[1] + 1
        elif x[i] <= ai[3]: ni[2] = ni[2] + 1
        elif x[i] <= ai[4]: ni[3] = ni[3] + 1
        elif x[i] <= ai[5]: ni[4] = ni[4] + 1
        else: ni[5] = ni[5] + 1

    for i in range(5):
        pi = np.append(pi, [sps.norm.cdf([ai[i + 1]], 0, 1) - sps.norm.cdf([ai[i]], 0, 1)])

    ans = np.array([], dtype = float)
    print('ni: ', *ni, np.sum(ni))
    print('pi: ',*pi, np.sum(pi))
    for i in range(5): pi[i] = pi[i] * n
    print('npi: ',*pi, np.sum(pi))
    for i in range(5): ni[i] = ni[i] - pi[i]
    print('ni - npi: ',*ni, np.sum(ni))
    for i in range(5): ans = np.append(ans, [ni[i]*ni[i] / pi[i]])
    print('ans: ',*ans, np.sum(ans))

def Step4():

    x1 = np.random.normal(loc = 0, scale = 1, size = 20)
    x2 = np.random.normal(loc = 0, scale = 1, size = 100)

    t1 = 2.093
    X11 = 32.8523
    X12 = 8.9065

    t2 = 1.9842
    X21 = 128.422
    X22 = 73.3611

    x = np.linspace(-3, 3, num = 100)
    y = sps.norm.pdf(x)

    #Нормальные интервальные оценки на основе точечных оценок
    mu = np.mean(x1)
    sigma = np.var(x1)

    print('n=20')
    print(mu - sigma*t1/np.sqrt(19), mu + sigma*t1/np.sqrt(19), sigma*np.sqrt(20)/np.sqrt(X11), sigma*np.sqrt(20)/np.sqrt(X12))

    mu = np.mean(x2)
    sigma = np.var(x2)
    print('n=100')
    print(mu - sigma*t2/np.sqrt(99), mu + sigma*t2/np.sqrt(99), sigma*np.sqrt(100)/np.sqrt(X21), sigma*np.sqrt(100)/np.sqrt(X22))

    print()
    print()
    #классические интервальные оценки на основе статистик
    u = 1.96

    n = 20
    U = u*np.sqrt((math.exp(1) + 2)/n)
    mu = np.mean(x1)
    sigma = np.var(x1)
    print('n=20')
    print(mu - sigma*u/np.sqrt(n), mu + sigma*u/np.sqrt(n), sigma*(1-U/2), sigma*(1+U/2))

    n = 100
    U = u*np.sqrt((math.exp(1) + 2)/n)
    mu = np.mean(x2)
    sigma = np.var(x2)
    print('n=100')
    print(mu - sigma*u/np.sqrt(n), mu + sigma*u/np.sqrt(n), sigma*(1-U/2), sigma*(1+U/2))


def main():
    #Step1()
    #Step2()
    #Step3()
    Step4()

if __name__ == "__main__":
    main()