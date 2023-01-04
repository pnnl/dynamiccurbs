# MPC for SeaTac controller
using JuMP
using DataFrames
using CSV
#using Gurobi
using Mosek
using MosekTools

A=[0.836317    0.113747   0.0         0.0
  -0.00685744  0.967017   0.0         0.0
   0.0         0.0        0.915093    0.177147
   0.0         0.0       -0.00108976  0.977752];  # A matrix

    B=[ -3.58394       2.03662     44.5946    0.0
        -0.000153837   1.76967      3.53123   0.0
        -5.75672       6.58173      0.0      12.8827
         1.14726      -0.00432909   0.0       2.10077];  # B matrix


# reading speed flow data from excel file as DataFrame
SF_data=CSV.read("seatac_vms1_vms2_treatment_ctl.csv", DataFrame)
volume_data=SF_data[1:end-1,17:18]; # inputs (DV;AV) at time t (departure volume; arrival volume)
IP=5637; # initial time
I_state=SF_data[IP,2:5]; # chosen initial state (congestion at 8)
Tf=16 # time-horizon (each step is 15 minute slot)
num_in=2 # number of control inputs (use arrival, use departure)
num_ext=2 # number of external inputs (departure volume, arrival volume)
num_state=4 # number of states in the model
crit_dep=30 # critical speed for departure
crit_arr=35 # critical speed for arrivals
VD=volume_data[IP:IP+Tf-1,:]'; # volume inputs in the interval
JM=Model(Mosek.Optimizer) # defining the JuMP model

# Defining variables of optimization problem
@variable(JM,U[1:num_in,1:Tf], Bin)  # control actions (use arrival, use departure--binary)
@variable(JM,X[1:num_state, 1:Tf+1])  # state variable
@variable(JM,Crit_arr_vel[1:Tf])   # minimum of 1 and arrival speed
@variable(JM,Crit_dep_vel[1:Tf])  # minimum of 1 and departure speed

#Defining constraints of the optimization problem
@constraint(JM, X .>= 0) # speed or flow cannot be negative
for i=1:Tf
   @constraint(JM, sum(U[:,i]) <= 1) # at most one action every time-step
end
for i=1:num_state
   @constraint(JM, X[i,1] .== I_state[i]) # defining initial state of the system
end
for i=1:Tf
   @constraint(JM, X[:,i+1] .== A*X[:,i] + B[:,1:2]*U[:,i] + B[:,3:4]*VD[:,i]) # state transition model
end
@constraint(JM,Crit_dep_vel .<= 1)
@constraint(JM,Crit_dep_vel .<= X[2,2:Tf+1]/crit_dep)
@constraint(JM,Crit_arr_vel .<= 1)
@constraint(JM,Crit_arr_vel .<= X[4,2:Tf+1]/crit_dep)

#objective is to keep critical speed (departure and arrival) around 1
@objective(JM,Min,sum(((Crit_dep_vel)-ones(Tf)).^2) + sum(((Crit_arr_vel)-ones(Tf)).^2))

status_SeaTac=optimize!(JM) # solve the optimization problem

X_value=value.(X)
U_value=value.(U)
