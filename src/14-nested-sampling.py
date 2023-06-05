import numpy as np

sigma_x = 0.1
sigma_y = 0.5

xmin = -0.5
xmax =  0.5
ymin = -0.5
ymax =  0.5

def f(x, y, sigma_x, sigma_y):
    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-x ** 2 / (2 * sigma_x ** 2) - y ** 2 / (2 * sigma_y ** 2))

# prior is uniform

M = 1000
theta = []

for i in range(M):
    aux = []
    for j in range(2):
        aux.append(np.random.uniform(xmin, xmax))
    theta.append(aux)


L = []
# likelihood
for i in range(M):
    L.append(f(theta[i][0], theta[i][1], sigma_x, sigma_y))

# prior mass
X = []
t = []

X.append(1)

posterior = 1
k = 0
Z = 0

while(posterior > 0.01):

    min_index = L.index(min(L))
    Lmin = L[min_index]
    theta_min = theta[min_index]

    print("Smallest likelihood: " + str(Lmin))

    t.append(np.random.beta(a = M, b = 1))

    if k > 0:
        X.append(X[-1] * t[-1])

        # evidence
        Z += (X[-2] - X[-1]) * Lmin

    print("Current prior mass: " + str(X[-1]))
    print("Current evidence: " + str(Z) + "\n")

    theta.pop(min_index)
    L.pop(min_index)

    posterior = Lmin * X[-1] / Z

    while(True):
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)

        if f(x, y, sigma_x, sigma_y) > Lmin:
            L.append(f(x, y, sigma_x, sigma_y))
            theta.append([x, y])
            break

    k += 1

print("The integral value is " + str(Z) + ", with the actual value being 0.68267.")