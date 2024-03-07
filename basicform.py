# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:03:58 2024

@author: busra
"""

import pandas as pd
import numpy as np
from gurobipy import *

m = Model("ENS491_first")

grid_cost_history = [3, 3, 4.7, 3.2, 4.7, 2.4, 3, 3, 4.7, 3.2, 4.7, 2.4, 3, 3, 4.7, 3.2, 4.7, 2.4, 3, 3, 4.7, 3.2, 4.7, 2.4]

P = [5, 8, 6, 4, 4, 2, 3, 7, 2, 8, 1, 5, 5, 3, 2, 8, 6, 5, 9, 10, 5, 2, 3, 4]

demand_history = [5, 7, 5, 4, 17, 4, 8, 16, 3, 7, 12, 8, 7, 2, 15, 3, 4, 3, 9, 20, 8, 13, 6, 7]

distributor = 1
customers = 1
time = 24     


X = m.addVars(distributor, customers, time, lb=0, vtype= GRB.CONTINUOUS, name = "x")
E = m.addVars(distributor, customers, time, lb=0, vtype= GRB.CONTINUOUS, name = "e")
#Y = m.addVars(distributor, customers, time, lb=0, ub=1, vtype= GRB.BINARY, name = "y")
I = m.addVars(time, lb=0, vtype= GRB.CONTINUOUS, name = "I")



m.addConstrs(I[t] + P[t] == quicksum(X[i,j,t] for i in range (distributor) for j in range(customers)) + I[t+1] for t in range(time-1))
m.addConstr(I[0] == 0)
m.addConstr(I[time-1] == 0, "FinalInventory")


m.addConstrs(X[i,j,t] <= P[t] + I[t] for i in range(distributor) for j in range(customers) for t in range(time))

m.addConstrs((quicksum(X[i,j,t] + E[i,j,t] for i in range(distributor) for j in range(customers)) == demand_history[t] for t in range(time)), "ExactDemand")



m.setObjective(quicksum(grid_cost_history[t] * E[i, j, t] for t in range(time) for i in range(distributor) for j in range(customers)), GRB.MINIMIZE)


#m.setObjective(quicksum(grid_cost_history[t]* E[i,j,t] for t in range(time)) + quicksum(Y[i,j,t]*distance[i][j]), GRB.MINIMIZE)



m.update()
# Optimize model
m.optimize()


# Print solution
print('Optimal solution:')
for v in m.getVars():
    print(f"{v.varName} = {v.x}")


# Print optimal objective value
print('Optimal objective value: ' + str(m.objVal) + "\n")


