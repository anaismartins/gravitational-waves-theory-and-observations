import numpy as np
import scipy.integrate as integrate
import sympy as sym
from sympy import exp as sym_exp
from sympy import symbols, I, conjugate, lambdify
from numpy import pi
import random

G = 6.67430e-11 # m3 kg-1 s-2
msun = 1.989e30 # kg
Gsun = G * 1.989e30

c = 3e8 # m s-1

fs = 20 # Hz
S0 = 1e-49 # Hz-1

S0 = 10e-49 # Hz-1

SNRopt = 32.4

theta = symbols('f A Mc eta tc Phiref')
theta = list(theta)

phi0 = 1
phi1 = 0
phi2 = 20 / 9 * (743 / 336 + 11 * theta[3] / 4)
phi3 = -16 * pi
phi4 = 10 * (3058673 / 1016064 + 5429 * theta[3] / 1008 + 617 * theta[3] ** 2 / 144)

phi = [phi0, phi1, phi2, phi3, phi4]

vc = ((pi * Gsun * theta[2] * theta[0]) / (theta[3] ** (3/5) * c ** 3)) ** (1/3)

Psi = 2 * pi * theta[0] * theta[4] - theta[5]
for k in range(0, 5):
    Psi = Psi + 3 / (128 * theta[3] * vc ** 5) * (phi[k] * vc ** k)

h = theta[1] * theta[0] ** (-7/6) * sym_exp(I * Psi)

Sn = S0 * ((theta[0] / 215) ** -4.14 - 5 * (theta[0] / 215) ** -2 + 111 * (1 - (theta[0] / 215) ** 2 + (theta[0] / 215) ** 4 / 2) / (1 + (theta[0] / 215) ** 2 / 2))

m1 = 1.4 # Msun
m2 = m1

Mc = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1/5)
eta = m1 * m2 / (m1 + m2) ** 2

tc = 0
Phiref = 0

fmin = 20
fmax = c ** 3 / (6 ** (3/2) * pi * Gsun * (m1 + m2))

def SNR(subst):
    snr_int = conjugate(h) * h / Sn

    snr = lambdify(theta[0], snr_int.subs(subst))

    snr_opt = sym.sqrt(4 * np.real(integrate.quad(snr, fmin, fmax)[0]))
    return snr_opt

best_A = 0
best_snr = 0

snr_not_found = True
iterations = 0
maxA = 10e-20
minA = 0

while snr_not_found:
    iterations = iterations + 1

    A = random.uniform(minA, maxA)
    subst = [(theta[1], A), (theta[2], Mc), (theta[3], eta), (theta[4], tc), (theta[5], Phiref)]

    snr_opt = SNR(subst)
    if abs(SNRopt - snr_opt) < abs(SNRopt - best_snr):
        best_A = A
        best_snr = snr_opt

    if (snr_opt > SNRopt):
        maxA = A

    if (snr_opt < SNRopt):
        minA = A

    print("At iteration " + str(iterations) + ", SNR is " + str(snr_opt) + " and A is " + str(A))

    if abs(best_snr - SNRopt) < 0.01:
        snr_not_found = False

    if iterations > 100:
        # max amount of iterations to try for
        snr_not_found = False

with open("answers.txt", "w") as file:
    file.write("The closest I could get to the optimal SNR, " + str(SNRopt) + ", was " + str(best_snr) + " with A = " + str(best_A) + "\n\n")

print("The closest I could get to the optimal SNR, " + str(SNRopt) + ", was " + str(best_snr) + " with A = " + str(best_A) + "\n")

def integrand(i, j):
    return conjugate(h.diff(theta[i])) * h.diff(theta[j]) / Sn

subst = [(theta[1], best_A), (theta[2], Mc), (theta[3], eta), (theta[4], tc), (theta[5], Phiref)]

def fisher(i, j, subst):
    integr = integrand(i,j)

    # \int a + ib = \int a + i \int b so we can take out the imaginary part :)
    integr_re = sym.re(integr)
    #integr_im = sym.im(integr)

    integr_re = lambdify(theta[0], integr_re.subs(subst))
    #integr_im = lambdify(theta[0], integr_im.subs(subst))

    #return 4 * np.real(integrate.quad(integr_re, fmin, fmax)[0] + 1j*integrate.quad(integr_im, fmin, fmax)[0])
    return 4 * np.real(integrate.quad(integr_re, fmin, fmax)[0])

fisher_matrix = []

for i in range (1, 6):
    aux = []
    for j in range(1, 6):
        aux.append(fisher(i, j, subst))
    if i == 0:
        fisher_matrix[0] = aux
    else:       
        fisher_matrix.append(aux)

print("The full fisher matrix with A included is\n")

with open("answers.txt", "a") as file:
    file.write("The full fisher matrix with A included is\n")
    for i in range(0, 5):
        for j in range(0, 5):
            file.write("{:e}".format(fisher_matrix[i][j]) + " ")
            print("{:e}".format(fisher_matrix[i][j]), end=" ")
        file.write("\n")
        print("\n", end="")

    file.write("\nBut like we found out in exercise b, Gamma_0i and Gamma_i0, with i <> 0, are all zeros, so we can/will use only Gamma_ij, with i, j <> 0: ")

print("\nBut like we found out in exercise b, Gamma_0i and Gamma_i0, with i <> 0, are all zeros, so we can/will use only Gamma_ij, with i, j <> 0: ")

crop_fisher = []

for i in range(1, 5):
    aux = []
    for j in range(1, 5):
        aux.append(fisher_matrix[i][j])
    crop_fisher.append(aux)

with open("answers.txt", "a") as file:
    for i in range(0, 4):
        for j in range(0, 4):
            file.write("{:e}".format(crop_fisher[i][j]) + " ")
            print("{:e}".format(crop_fisher[i][j]), end=" ")
        file.write("\n")
        print("\n", end="")

Sigma = np.linalg.inv(crop_fisher)

print("\nThe Sigma matrix is\n")

with open("answers.txt", "a") as file:
    file.write("\nThe Sigma matrix is" + "\n")
    for i in range(0, 4):
        for j in range(0, 4):
            file.write("{:e}".format(Sigma[i][j]) + " ")
            print("{:e}".format(Sigma[i][j]), end=" ")
        file.write("\n")
        print("\n", end="")

sqrtSigma = np.sqrt(Sigma)

deltaMcMc = sqrtSigma[0][0]/Mc
deltaeta = sqrtSigma[1][1]
deltatc = sqrtSigma[2][2]
deltaPhiref = sqrtSigma[3][3]

print("\ndelta Mc / Mc = " + "{:e}".format(deltaMcMc))
print("delta eta = " + "{:e}".format(deltaeta))
print("delta tc = " + "{:e}".format(deltatc))
print("delta Phiref = " + "{:e}".format(deltaPhiref))