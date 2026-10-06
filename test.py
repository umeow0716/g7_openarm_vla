from g7_openarm_pinnzoo import PinnZooModel, mass_matrix, zero_state


model = PinnZooModel()
x = zero_state(model)
"""
#  [
x y z
x y z

8

 
    
]
"""

M = mass_matrix(model, x)[14:14+7, 14:14+7]

# print it as square

for i in range(7):
    print(M[i, :])
