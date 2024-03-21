import pandas as pd
import numpy as np
from gurobipy import *

# Load and preprocess the data
demand_history_df = pd.read_csv('/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/averaged_predictions_2017.csv')
solar_production_df = pd.read_csv('/Users/yunustopcu/Documents/GitHub/predictive-solar-supply-chain/hourly_solar_production_data.csv')

# Ensure date formats are consistent and align the datasets
demand_history_df['Date'] = pd.to_datetime(demand_history_df['Date'])
solar_production_df['date'] = pd.to_datetime(solar_production_df['date'])
demand_history = demand_history_df['Average_Prediction'].tolist()
P = solar_production_df['GTI'].tolist()

# Constants and parameters
grid_cost_history = [2.2, 2.2, 1.9, 1.9, 1.8, 1.9, 1.7, 1.8, 2.2, 2.3, 2.5, 2.8, 2.9, 3.0, 3.0, 3.2, 3.2, 6.4, 3.2, 3.1, 2.8, 2.8, 2.5, 2.2]
M = 9999
distributor = 1
customers = 1
time = 24
initial_inventory = 0

# Iterate over each day in the dataset
for day_start in range(0, len(demand_history), time):
    m = Model(f"ENS491_first_Day_{day_start // time + 1}")

    # Adjust the index for P and demand_history to match the current day
    P_day = P[day_start:day_start+time]
    demand_day = demand_history[day_start:day_start+time]

    # Define variables
    X = m.addVars(distributor, customers, time, lb=0, vtype=GRB.CONTINUOUS, name="x")
    E = m.addVars(distributor, customers, time, lb=0, vtype=GRB.CONTINUOUS, name="e")
    Y = m.addVars(distributor, customers, time, lb=0, ub=1, vtype=GRB.BINARY, name="y")
    I = m.addVars(time, lb=0, vtype=GRB.CONTINUOUS, name="I")

    # Constraints
    m.addConstr(I[0] == initial_inventory)
    m.addConstrs(I[t] + P_day[t] == quicksum(X[i,j,t] for i in range(distributor) for j in range(customers)) + I[t+1] for t in range(time-1))
    m.addConstr(I[time-1] == 0, "FinalInventory")

    m.addConstrs(X[i,j,t] <= P_day[t] + I[t] for i in range(distributor) for j in range(customers) for t in range(time))
    m.addConstrs((quicksum(X[i,j,t] + E[i,j,t] for i in range(distributor) for j in range(customers)) == demand_day[t] for t in range(time)), "ExactDemand")
    m.addConstrs(X[i,j,t] <= Y[i,j,t]*M for i in range(distributor) for j in range(customers) for t in range(time))

    # Objective
    m.setObjective(quicksum(grid_cost_history[t] * E[i, j, t] for t in range(time) for i in range(distributor) for j in range(customers)), GRB.MINIMIZE)

    # Optimize model for the current day
    m.optimize()

    # Update initial inventory for the next day
    if m.status == GRB.OPTIMAL:
        initial_inventory = I[time-1].X  # Final inventory of the current day becomes the initial inventory for the next day

        # Optionally, print solution for each day
        print(f'MODEL HAS BEEN SOLVED FOR DAY {day_start // time + 1}')
        for v in m.getVars():
            if v.X > 0:
                print(f"{v.varName} = {v.X}")
        print('Optimal objective value: ' + str(m.objVal) + "\n")
    else:
        print(f"Model for Day {day_start // time + 1} is infeasible.")
