# MPC for SeaTac controller
using JuMP
using DataFrames
using CSV
#using Gurobi
using Mosek
using MosekTools
using LinearAlgebra
using MathOptInterface


A=[0.836317    0.113747   0.0         0.0
  -0.00685744  0.967017   0.0         0.0
   0.0         0.0        0.915093    0.177147
   0.0         0.0       -0.00108976  0.977752];  # A matrix

    B=[ -3.58394       2.03662     44.5946    0.0
        -0.000153837   1.76967      3.53123   0.0
        -5.75672       6.58173      0.0      12.8827
         1.14726      -0.00432909   0.0       2.10077];  # B matrix


# reading speed flow data from excel file as DataFrame
SF_data=CSV.read("seatac_vms1_vms2_treatment_vol.csv", DataFrame)
volume_data=SF_data[1:end-1,17:18]; # inputs (DV;AV) at time t (departure volume; arrival volume)
IP=6009; # initial time
Tf=16 # time-horizon (each step is 15 minute slot)
To=25 # receding horizon length
num_in=2 # number of control inputs (use arrival, use departure)
num_ext=2 # number of external inputs (departure volume, arrival volume)
num_state=4 # number of states in the model
crit_dep=30 # critical speed for departure
crit_arr=35 # critical speed for arrivals
X_final=zeros(num_state,To); # storing output state values over time
U_final=zeros(num_in,To);   # storing control inputs over time

for run_no=1:To   # running a loop over the receding horizon

   I_state=SF_data[IP,3:6]; # choosing initial state at time IP
   I_state=Array(I_state); # converting from DataFrame to array
   if run_no>1             # for next steps using the previous obtained state in the loop
      I_state=X_final[:,run_no-1];
   end

   VD1=volume_data[IP+run_no-1:IP+run_no-1+Tf-1,:]; # volume inputs in the interval
   VD=Matrix(VD1)'
   JM=Model(Mosek.Optimizer) # defining the JuMP model

   # Defining variables of optimization problem
   @variable(JM,U[1:num_in,1:Tf], Bin)  # control actions (use arrival, use departure--binary)
   @variable(JM,X[1:num_state, 1:Tf+1])  # state variable
   @variable(JM,Crit_arr_vel[1:Tf])   # minimum of 1 and arrival speed
   @variable(JM,Crit_dep_vel[1:Tf])  # minimum of 1 and departure speed

   #Defining constraints of the optimization problem
   @constraint(JM, X .>= 0) # speed or flow cannot be negative
   @constraint(JM, X[1,:] .<= 475) # upper limit on departure flow
   @constraint(JM, X[2,:] .<= 49.5) # upper limit on departure speed
   @constraint(JM, X[3,:] .<= 445) # upper limit on arrival flow
   @constraint(JM, X[4,:] .<= 67.5) # upper limit on arrival speed
   for i=1:Tf
      @constraint(JM, sum(U[:,i]) <= 1) # at most one action every time-step
   end
   for i=1:num_state
      @constraint(JM, X[i,1] == I_state[i]) # defining initial state of the system
   end
   for i=1:Tf
      @constraint(JM, X[:,i+1] .== A*X[:,i] + B[:,1:2]*U[:,i] + B[:,3:4]*VD[:,i]) # state transition model
   end
   @constraint(JM,Crit_dep_vel .<= 1)
   @constraint(JM,Crit_dep_vel .<= X[2,2:Tf+1]/crit_dep)
   @constraint(JM,Crit_arr_vel .<= 1)
   @constraint(JM,Crit_arr_vel .<= X[4,2:Tf+1]/crit_dep)

   #objective is to keep critical speed (departure and arrival) around 1
   @objective(JM,Min,sum(((Crit_dep_vel)-ones(Tf)).^2) + sum(((Crit_arr_vel)-ones(Tf)).^2)+0.0001*sum(sum(U)))

   status_SeaTac=optimize!(JM) # solve the optimization problem

   X_value=value.(X)  # solved state value
   U_value=value.(U)  # solved input value

   X_final[:,run_no]=X_value[:,2]; # storing values
   U_final[:,run_no]=U_value[:,1];

end
