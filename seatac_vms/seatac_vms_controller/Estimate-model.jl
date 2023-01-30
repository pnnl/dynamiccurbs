using JuMP
using DataFrames
using CSV
#using Gurobi
using Mosek
using MosekTools
using LinearAlgebra
using MathOptInterface
#using Ipopt

SF_data=CSV.read("seatac_vms1_vms2_treatment_vol.csv", DataFrame) # reading data from excel file

X1=SF_data[1:end-1,3:6]; # state (DF;DS;AF;AS) at time t
X2=SF_data[2:end,3:6]; # state (DF;DS;AF;AS) at time t+1
U1=SF_data[1:end-1,11:12]; # inputs (TA;TD) at time t
U2=SF_data[1:end-1,17:18]; # inputs (DV;AV) at time t (departure volume; arrival volume)
X1=Matrix(X1); # converting to matrix
X2=Matrix(X2);
U1=Matrix(U1);
U2=Matrix(U2);

# X1=reverse(X1, dims=1)
# X2=reverse(X2, dims=1)
# U1=reverse(U1, dims=1)
# U2=reverse(U2, dims=1)

Y1=[X1';U1';U2']; # concatenating inputs

JM=Model(Mosek.Optimizer) # defining the JuMP model

        # Defining variables of optimization problem
@variable(JM,At[1:4, 1:8])  # state variable + input matrix
@variable(JM,t[1:11912])  # auxillary variable for norm
@variable(JM,q[1:4])   # auxillary variable for regularization term
@variable(JM,maxnorm)
for i=1:11912
    @constraint(JM, [t[i]; At*Y1[:,i]-X2[i,:]] in SecondOrderCone())
end
        #Constraints on the A and B matrices (from model information)
@constraint(JM, At[1,3] == 0) # effect of arrival flow on departure flow
@constraint(JM, At[1,4] == 0) # effect of arrival speed on departure flow
@constraint(JM, At[2,3] == 0) # effect of arrival flow on departure speed
@constraint(JM, At[2,4] == 0) # effect of arrival speed on departure speed
@constraint(JM, At[3,1] == 0) # effect of departure flow on arrival flow
@constraint(JM, At[3,2] == 0) # effect of departure speed on arrival flow
@constraint(JM, At[4,1] == 0) # effect of departure flow on arrival speed
@constraint(JM, At[4,2] == 0) # effect of departure speed on arrival speed

@constraint(JM, At[1,8] == 0) # effect of arrival volume on departure flow
@constraint(JM, At[2,8] == 0) # effect of arrival volume on departure speed
@constraint(JM, At[3,7] == 0) # effect of departure volume on arrival flow
@constraint(JM, At[4,7] == 0) # effect of departure volume on arrival speed

@constraint(JM, At[2,5] <= 0) # effect of treat arrival on departure speed (-ve)
@constraint(JM, At[2,6] >= 0) # effect of treat departure on departure speed (+ve)
@constraint(JM, At[4,5] >= 0) # effect of treat arrival on arrival speed (+ve)
@constraint(JM, At[4,6] <= 0) # effect of treat departure on arrival speed (-ve)

@constraint(JM, [maxnorm; vec(At)] in MathOptInterface.NormSpectralCone(4, 8))

for j=1:4
    @constraint(JM, [q[j]; At[j,:]] in SecondOrderCone())
end
# objective is to minimize fit of Y1 to X2 using At matrix ,i.e., ||X2-At*Y1||_2 + regularization term on At
@objective(JM,Min,sum(t.^2)+1e2*sum(q))

status=optimize!(JM) # solve the optimization problem

A_value=value.(At)
A=A_value[:,1:4]; # A matrix
B=A_value[:,5:8]; # B matrix
t_value=value.(t); # fit error

# calculating modeling error
Er=zeros(Float64,11912,4);
for i=1:11912
    Er[i,:]=X2[i,:]-A_value*Y1[:,i];
end
