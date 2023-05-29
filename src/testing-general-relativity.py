import numpy as np
import scipy.integrate as integrate
import sympy as sym
from sympy import exp as sym_exp
from sympy import symbols, I, conjugate, lambdify
from numpy import pi
import random

G = 6.67430e-11  # m3 kg-1 s-2
msun = 1.989e30  # kg
Gsun = G * msun

c = 3e8  # m s-1

S0 = 1e-49  # Hz-1

SNRopt = 32.4

print("Exercise c)")

with open("answers.txt", "w") as file:
    file.write("Exercise c)\n")

f = symbols("f")
theta = symbols("A Mc eta tc Phiref")
theta = list(theta)

phi0 = 1
phi1 = 0
phi2 = 20 / 9 * (743 / 336 + 11 * theta[2] / 4)
phi3 = -16 * pi
phi4 = 10 * (3058673 / 1016064 + 5429 * theta[2] / 1008 + 617 * theta[2] ** 2 / 144)

phi = [phi0, phi1, phi2, phi3, phi4]

vc = ((pi * Gsun * theta[1] * f) / (theta[2] ** (3 / 5) * c**3)) ** (1 / 3)

Psi = 2 * pi * f * theta[3] - theta[4]
for k in range(0, 5):
    Psi = Psi + 3 / (128 * theta[2] * vc**5) * (phi[k] * vc**k)

h = theta[0] * f ** (-7 / 6) * sym_exp(I * Psi)

Sn = S0 * (
    (f / 215) ** -4.14
    - 5 * (f / 215) ** -2
    + 111 * (1 - (f / 215) ** 2 + (f / 215) ** 4 / 2) / (1 + (f / 215) ** 2 / 2)
)

m1 = 1.4  # Msun
m2 = m1

Mc = (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
eta = m1 * m2 / (m1 + m2) ** 2

tc = 0
Phiref = 0

fmin = 20
fmax = c**3 / (6 ** (3 / 2) * pi * Gsun * (m1 + m2))


def SNR(subst):
    snr_int = conjugate(h) * h / Sn

    snr = lambdify(f, snr_int.subs(subst))

    snr_opt = sym.sqrt(4 * integrate.quad(snr, fmin, fmax)[0])
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
    subst = [
        (theta[0], A),
        (theta[1], Mc),
        (theta[2], eta),
        (theta[3], tc),
        (theta[4], Phiref),
    ]

    snr_opt = SNR(subst)
    if abs(SNRopt - snr_opt) < abs(SNRopt - best_snr):
        best_A = A
        best_snr = snr_opt

    if snr_opt > SNRopt:
        maxA = A

    if snr_opt < SNRopt:
        minA = A

    print(
        "At iteration "
        + str(iterations)
        + ", SNR is "
        + f"{snr_opt:.2f}"
        + " and A is "
        + f"{A:.2}"
    )

    if abs(best_snr - SNRopt) < 0.01:
        snr_not_found = False

    if iterations > 100:
        # max amount of iterations to try for
        snr_not_found = False

with open("answers.txt", "a") as file:
    file.write(
        "The closest I could get to the optimal SNR, "
        + str(SNRopt)
        + ", was "
        + f"{best_snr:.2f}"
        + " with A = "
        + f"{best_A:.2}"
        + "\n\n"
    )

print(
    "The closest I could get to the optimal SNR, "
    + str(SNRopt)
    + ", was "
    + f"{best_snr:.2f}"
    + " with A = "
    + f"{best_A:.2}"
    + "\n"
)


def integrand(h, i, j):
    return conjugate(h.diff(theta[i])) * h.diff(theta[j]) / Sn


subst = [
    (theta[0], best_A),
    (theta[1], Mc),
    (theta[2], eta),
    (theta[3], tc),
    (theta[4], Phiref),
]


def fisher(h, i, j, subst):
    integr = integrand(h, i, j)

    # \int a + ib = \int a + i \int b so we can take out the imaginary part :)
    integr_re = sym.re(integr)
    integr_re = lambdify(f, integr_re.subs(subst))
    return 4 * np.real(integrate.quad(integr_re, fmin, fmax)[0])


fisher_matrix = []

for i in range(0, 5):
    aux = []
    for j in range(0, 5):
        aux.append(fisher(h, i, j, subst))

    fisher_matrix.append(aux)

print("The full fisher matrix with A included is\n")

with open("answers.txt", "a") as file:
    file.write("The full fisher matrix with A included is\n")
    for i in range(0, 5):
        for j in range(0, 5):
            file.write(f"{fisher_matrix[i][j]:.2}" + " ")
            print(f"{fisher_matrix[i][j]:.2}", end=" ")
        file.write("\n")
        print("\n", end="")

    file.write(
        "\nBut like we found out in exercise b, Gamma_0i and Gamma_i0, with i <> 0, are all zeros, so we can/will use only Gamma_ij, with i, j <> 0: "
    )

print(
    "\nBut like we found out in exercise b, Gamma_0i and Gamma_i0, with i <> 0, are all zeros, so we can/will use only Gamma_ij, with i, j <> 0: "
)

crop_fisher = []

for i in range(1, 5):
    aux = []
    for j in range(1, 5):
        aux.append(fisher_matrix[i][j])
    crop_fisher.append(aux)

with open("answers.txt", "a") as file:
    for i in range(0, 4):
        for j in range(0, 4):
            file.write(f"{crop_fisher[i][j]:.2}" + " ")
            print(f"{crop_fisher[i][j]:.2}", end=" ")
        file.write("\n")
        print("\n", end="")

Sigma = np.linalg.inv(crop_fisher)

print("\nThe Sigma matrix is\n")

with open("answers.txt", "a") as file:
    file.write("\nThe Sigma matrix is" + "\n")
    for i in range(0, 4):
        for j in range(0, 4):
            file.write(f"{Sigma[i][j]:.2}" + " ")
            print(f"{Sigma[i][j]:.2}", end=" ")
        file.write("\n")
        print("\n", end="")

sqrtSigma = np.sqrt(Sigma)

deltaMcMc = sqrtSigma[0][0] / Mc
deltaeta = sqrtSigma[1][1]
deltatc = sqrtSigma[2][2]
deltaPhiref = sqrtSigma[3][3]

with open("answers.txt", "a") as file:
    file.write("\ndelta Mc / Mc = " + f"{deltaMcMc:.2}\n")
    file.write("delta eta = " + f"{deltaeta:.2}\n")
    file.write("delta tc = " + f"{deltatc:.2}\n")
    file.write("delta Phiref = " + f"{deltaPhiref:.2}")

print("\ndelta Mc / Mc = " + f"{deltaMcMc:.2}")
print("delta eta = " + f"{deltaeta:.2}")
print("delta tc = " + f"{deltatc:.2}")
print("delta Phiref = " + f"{deltaPhiref:.2}")

# d ---------------------------------------------------------------------------------------------------

print("\n\nExercise d)\n")

with open("answers.txt", "a") as file:
    file.write("\n\nExercise d)\n")

# A is not used/needed anymore but I'll leave it here because otherwise I would have to change everything
theta = symbols("A Mc eta tc Phiref phi_2")
theta = list(theta)

Psi_new = Psi + theta[5] * vc ** (-2) * 3 / (128 * theta[2] * vc**5)
h = best_A * f ** (-7 / 6) * sym_exp(I * Psi_new)

phi_2 = 0

subst = [
    (theta[1], Mc),
    (theta[2], eta),
    (theta[3], tc),
    (theta[4], Phiref),
    (theta[5], phi_2),
]

fisher_matrix_new = []

for i in range(1, 6):
    aux = []
    for j in range(1, 6):
        aux.append(fisher(h, i, j, subst))

    fisher_matrix_new.append(aux)

print("The new fisher matrix (without A and with phi-2) is\n")

with open("answers.txt", "a") as file:
    file.write("The new fisher matrix (without A and with phi-2) is\n")
    for i in range(0, 5):
        for j in range(0, 5):
            file.write(f"{fisher_matrix_new[i][j]:.2}" + " ")
            print(f"{fisher_matrix_new[i][j]:.2}", end=" ")
        file.write("\n")
        print("\n", end="")

Sigma_new = np.linalg.inv(fisher_matrix_new)

print("\nThe new Sigma matrix is\n")

with open("answers.txt", "a") as file:
    file.write("\nThe new Sigma matrix is" + "\n")
    for i in range(0, 5):
        for j in range(0, 5):
            file.write(f"{Sigma_new[i][j]:.2}" + " ")
            print(f"{Sigma_new[i][j]:.2}", end=" ")
        file.write("\n")
        print("\n", end="")

deltaphi_2 = np.sqrt(Sigma_new[4][4])

with open("answers.txt", "a") as file:
    file.write(
        "\ndelta phi-2 = "
        + f"{deltaphi_2:.2}\n\nBinary pulsars are well-suited for this test of GR because we can measure them through radio telescopes and get very accurate measurements of masses, as well as their orbitals because they have strong gravitational fields in their vicinities, high orbital velocities and they are considered accurate clocks"
    )

print(
    "\ndelta phi-2 = "
        + f"{deltaphi_2:.2}\n\nBinary pulsars are well-suited for this test of GR because we can measure them through radio telescopes and get very accurate measurements of masses, as well as their orbitals because they have strong gravitational fields in their vicinities, high orbital velocities and they are considered accurate clocks"
)

# e -------------------------------------------------------------------------------------

print("\n\nExercise e)\n")

with open("answers.txt", "a") as file:
    file.write("\n\nExercise e)")

theta = symbols("A Mc eta tc Phiref deltaphi3")
theta = list(theta)

Psi_new2 = Psi + theta[5] * (-16) * pi * vc ** (3) * 3 / (128 * theta[2] * vc**5)
h = best_A * f ** (-7 / 6) * sym_exp(I * Psi_new2)

deltaphi3 = 0

subst = [
    (theta[1], Mc),
    (theta[2], eta),
    (theta[3], tc),
    (theta[4], Phiref),
    (theta[5], deltaphi3),
]

fisher_matrix_new = []

for i in range(1, 6):
    aux = []
    for j in range(1, 6):
        aux.append(fisher(h, i, j, subst))

    fisher_matrix_new.append(aux)

Sigma_new = np.linalg.inv(fisher_matrix_new)

Deltadeltaphi3 = np.sqrt(Sigma_new[4][4])

with open("answers.txt", "a") as file:
    file.write(
        "\nDelta delta phi 3 = "
        + f"{Deltadeltaphi3:.2}\n\nThe results obtained here are one order of magnitude lower than those of the mentioned paper."
    )

print(
    "Delta delta phi 3 = "
    + f"{Deltadeltaphi3:.2}\n\nThe results obtained here are one order of magnitude lower than those of the mentioned paper."
)

# f -------------------------------------------------------------------------------------

print("\n\nExercise f)\n")

with open("answers.txt", "a") as file:
    file.write("\n\nExercise f)\n")

theta = symbols("A Mc eta tc Phiref phi_2 deltaphi3")
theta = list(theta)

S0 = 1e-50  # Hz

Sn = (
    S0
    * (
        2.39e-27 * (f / 100) ** (-15.64)
        + 0.349 * (f / 100) ** -2.145
        + 1.76 * (f / 100) ** -0.12
        + 0.409 * (f / 100) ** 1.1
    )
    ** 2
)

fmin = 5  # Hz

Psi_new3 = Psi_new + theta[6] * (-16) * pi * vc ** (3) * 3 / (128 * theta[2] * vc**5)
h = best_A * f ** (-7 / 6) * sym_exp(I * Psi_new3)

subst = [
    (theta[1], Mc),
    (theta[2], eta),
    (theta[3], tc),
    (theta[4], Phiref),
    (theta[5], phi_2),
    (theta[6], deltaphi3),
]

new_snr = SNR(subst)

with open("answers.txt", "a") as file:
    file.write(
        "The new SNR is "
        + f"{new_snr:.1f}. This value is much higher than the original SNR, as expected, since the ET is meant to be a better detector than the ones we have now.\n"
    )

print(
    "The new SNR is "
    + f"{new_snr:.1f}. This value is much higher than the original SNR, as expected, since the ET is meant to be a better detector than the ones we have now."
)

fisher_matrix_new = []

for i in range(1, 7):
    aux = []
    for j in range(1, 7):
        aux.append(fisher(h, i, j, subst))

    fisher_matrix_new.append(aux)

Sigma_new = np.linalg.inv(fisher_matrix_new)

Deltaphi_2 = np.sqrt(Sigma_new[4][4])
Deltadeltaphi3 = np.sqrt(Sigma_new[5][5])

with open("answers.txt", "a") as file:
    file.write(
        "Delta phi -2 = "
        + f"{Deltaphi_2:.2}\nDelta delta phi 3 = "
        + f"{Deltadeltaphi3:.2}\n\n"
    )

print(
    "Delta phi -2 = "
    + f"{Deltaphi_2:.2}\nDelta delta phi 3 = "
    + f"{Deltadeltaphi3:.2}\n\n"
)
