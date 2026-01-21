import numpy as np


thetha_max = (180 / 180 * np.pi) / 2
k1 = 1.2989676453174887e-01
k2 = -2.2790603406357010e-02
k3 = -1.1560361547857329e-02
k4 = 3.6854238155112160e-03

x_pixel = 1920
y_pixel = 1536

# r_pixel = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
r_pixel = x_pixel

rd = thetha_max + k1 * thetha_max ** 3 + k2 * thetha_max ** 5 + k3 * thetha_max ** 7 + k4 * thetha_max ** 9

f = (r_pixel / 2) / rd

print(f)