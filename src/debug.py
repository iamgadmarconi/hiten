import numpy as np

map1 = np.array([[1, 0, 0, 0, 0, 0],
                [0, 1/np.sqrt(2), 0, 0, 1j/np.sqrt(2), 0],
                [0, 0, 1/np.sqrt(2), 0, 0, 1j/np.sqrt(2)],
                [0, 0, 0, 1, 0, 0],
                [0, 1j/np.sqrt(2), 0, 0, 1/np.sqrt(2), 0],
                [0, 0, 1j/np.sqrt(2), 0, 0, 1/np.sqrt(2)]], dtype=np.complex128) #  real = map1 @ complex

print(f"real = map1 @ complex:\n {map1}\n")

map2 = np.linalg.inv(map1) # complex = map2 @ real

print(f"complex = map2 @ real:\n {map2}\n")

# complex_6d = np.array([-9.99634542e-02+9.97984775e-19j,
#                        -8.55274617e-21+2.68934929e-04j,
#                        -6.60835552e-20+5.21039633e-01j,
#                        9.99634542e-02+1.19560352e-18j,
#                        -2.68934929e-04+9.39263070e-21j,
#                        5.21039633e-01-4.01479521e-20j])

# real_6d = map1 @ complex_6d # real = map1 @ complex

# print(f"real = map1 @ complex:\n {real_6d}\n")


initial_6d = np.array([0,0,0,0,0,0.72737742])

complex_6d = map2 @ initial_6d # complex = map2 @ real

print(f"complex_6d = map2 @ initial_6d:\n {complex_6d}\n")






