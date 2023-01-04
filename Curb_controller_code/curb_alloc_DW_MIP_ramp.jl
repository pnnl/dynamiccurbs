# Julia code for dynamic curb allocation comparing MIP and DW method (QCP constraint included)
# written by Nawaf
# curb allocation for Seattle (Insignia) without bus stop consideration
nb=26; # no of blocks
T=10; # no of timesteps
E=3; # no of curb types (1=paid parking, 2=commercial vehicles, 3=busus/public)

using CSV
using DataFrames
# read the curb length data
curb_data=CSV.read("./space_val_data/Iter2/ekey_length.csv", DataFrame)  # reading node data
tot_curbs=sum(curb_data[1:nb,3]);
loc=3; # no of demand locations
tot_T=96; # total time of data available
# read the space valuation of curbs for buses
curb_value_bus=zeros(Float64,tot_curbs,loc,tot_T);
curb_value_bus_tot=zeros(Float64,tot_curbs,tot_T);
global mp=0;
for i=1:nb
    curb_name=string(curb_data[i,2]);
    for k=1:loc
        loc_name=string(k);
        value_data=CSV.read(string("./space_val_data/Iter2/bus/",curb_name,"_loc_",loc_name,"_bus.csv"), DataFrame; header=false)
        for j=1:curb_data[i,3]
            for p=1:tot_T
                curb_value_bus[mp+j,k,p]=value_data[j,p];
            end
        end
    end
    global mp=mp+curb_data[i,3];
end
curb_value_bus_tot=reshape(sum(curb_value_bus,dims=2),tot_curbs,tot_T);
# read the space valuation of curbs for paid parking
curb_value_paid=zeros(Float64,tot_curbs,loc,tot_T);
curb_value_paid_tot=zeros(Float64,tot_curbs,tot_T);
global mp=0;
for i=1:nb
    curb_name=string(curb_data[i,2]);
    for k=1:loc
        loc_name=string(k);
        value_data=CSV.read(string("./space_val_data/Iter2/paid/",curb_name,"_loc_",loc_name,"_paid.csv"), DataFrame; header=false)
        for j=1:curb_data[i,3]
            for p=1:tot_T
                curb_value_paid[mp+j,k,p]=value_data[j,p];
            end
        end
    end
    global mp=mp+curb_data[i,3];
end
curb_value_paid_tot=reshape(sum(curb_value_paid,dims=2),tot_curbs,tot_T);
# read the space valuation of curbs for cv
curb_value_cv=zeros(Float64,tot_curbs,loc,tot_T);
curb_value_cv_tot=zeros(Float64,tot_curbs,tot_T);
global mp=0;
for i=1:nb
    curb_name=string(curb_data[i,2]);
    for k=1:loc
        loc_name=string(k);
        value_data=CSV.read(string("./space_val_data/Iter2/cv/",curb_name,"_loc_",loc_name,"_cv.csv"), DataFrame; header=false)
        for j=1:curb_data[i,3]
            for p=1:tot_T
                curb_value_cv[mp+j,k,p]=value_data[j,p];
            end
        end
    end
    global mp=mp+curb_data[i,3];
end
curb_value_cv_tot=reshape(sum(curb_value_cv,dims=2),tot_curbs,tot_T);

# read the distance (latitude and longitude) of curbs

dist_curb=zeros(Float64,tot_curbs,2);
global mp=0;
for i=1:nb
    curb_name=string(curb_data[i,2]);
    dist_data=CSV.read(string("./space_val_data/Iter2/distance/",curb_name,"_dist.csv"), DataFrame; header=false)
    for j=1:curb_data[i,3]
        for k=1:2
            dist_curb[mp+j,k]=dist_data[j,k];
        end
    end
    global mp=mp+curb_data[i,3];
end

# form the distance matrix
dist_matrix=zeros(Float64,tot_curbs,tot_curbs);
for i=1:tot_curbs
    for j=i+1:tot_curbs
        #if i!=j
            dist_matrix[i,j]=(1e-5)/((dist_curb[i,1]-dist_curb[j,1])^2+(dist_curb[i,2]-dist_curb[j,2])^2);
            #dist_matrix[j,i]=(1e-5)/((dist_curb[i,1]-dist_curb[j,1])^2+(dist_curb[i,2]-dist_curb[j,2])^2);
        #else
            #dist_matrix[i,j]=1;
        #end
    end
end


 b=100; # no of changes in allocations between timesteps
 # tmin=[120; 120; 120]; # minimum allocations over time
 # tmax=[1500; 1500; 1500]; # maximum allocations over time
t1min=[50; 50; 50]; # minimum allocations at one timestep (of a type of allocation)
t1max=[125; 125; 125]; # maximum allocations at one timestep (of a type of allocation)

# optimization problem starts
start_time=49;  # start the optimization horizon at this timestep (from tot_T=96)
using JuMP
using Gurobi
#using Mosek
#using MosekTools
JM=Model(with_optimizer(Gurobi.Optimizer, MIPGap=.005, OutputFlag=0)) # silent run
#JM=Model(Gurobi.Optimizer, MIPGap=.01)
# defining variables
@variable(JM,U[1:tot_curbs,1:T,1:E], Bin) # allocation variable
#(1=paid parking, 2=commercial vehicles, 3=busus/public)
@variable(JM,x[1:tot_curbs,1:T-1,1:E], Int) # change in allocation between timesteps
@variable(JM,w[1:T,1:E]) # regularization term for spacing
# defining objective
@objective(JM,Min,-sum(sum(curb_value_paid_tot[:,start_time+1:start_time+T].*U[:,:,1]))-sum(sum(curb_value_cv_tot[:,start_time+1:start_time+T].*U[:,:,2]))-sum(sum(curb_value_bus_tot[:,start_time+1:start_time+T].*U[:,:,3])))
 # maximize linear revenue+ regularization for spacing (neglect regularization for now)
# defining constraints
for i=1:tot_curbs
    for t=1:T
        @constraint(JM,sum(U[i,t,:])==1) # at a time and location only one allocation allowed
    end
end
for t=1:T-1
    for i=1:tot_curbs
        for e=1:E
            @constraint(JM,x[i,t,e] .== U[i,t+1,e]-U[i,t,e])
        end
    end
    @constraint(JM,sum(sum(x[:,t,:].^2))<=2*b) # limiting change in allocation between timesteps
end
# maximum and minimum no of allocations over time for each type of allocation
for e=1:E
    #@constraint(JM,sum(sum(U[:,:,e]))<=tmax[e])
    #@constraint(JM,sum(sum(U[:,:,e]))>=tmin[e])
end
# maximum and minimum no of allocations at each timestep for each allocation

for e=1:E
    for t=1:T
        @constraint(JM,sum(U[:,t,e])<=t1max[e])
        @constraint(JM,sum(U[:,t,e])>=t1min[e])
    end
end
# regularization distance is positive
@constraint(JM,w .>= 0)
#@constraint(JM,U .>= 0)
# maximize distance between same type allocations
for e=1:E
    for t=1:T
        @constraint(JM,U[:,t,e]'*dist_matrix*U[:,t,e]<=w[t,e])  # calculate regularization term
    end
end

status_1=optimize!(JM) # run the optimization
#
# for i=1:289
#     if value.(U[i,2,1]) ==1
#         println("paid")
#     elseif value.(U[i,2,2]) ==1
#         println("cv")
#     else
#         println("bus")
#     end
# end
U_final=value.(U);  # MIP solution
# dfU=DataFrame(Any[U_final[:,i,j] for j=1:E for i=1:T],:auto)
# CSV.write("U_final.csv",dfU)
# dfPaid=DataFrame(curb_value_paid_tot[:,50:59],:auto)
# CSV.write("paid_objective.csv",dfPaid)
# dfCV=DataFrame(curb_value_cv_tot[:,50:59],:auto)
# CSV.write("cv_objective.csv",dfCV)
# dfBus=DataFrame(curb_value_bus_tot[:,50:59],:auto)
# CSV.write("bus_objective.csv",dfBus)

println("MIP optimal solution")
println(sum(sum(curb_value_paid_tot[:,start_time+1:start_time+T].*value.(U[:,:,1])))+sum(sum(curb_value_cv_tot[:,start_time+1:start_time+T].*value.(U[:,:,2])))+sum(sum(curb_value_bus_tot[:,start_time+1:start_time+T].*value.(U[:,:,3]))))
MIP_sol_opt=sum(sum(curb_value_paid_tot[:,start_time+1:start_time+T].*value.(U[:,:,1])))+sum(sum(curb_value_cv_tot[:,start_time+1:start_time+T].*value.(U[:,:,2])))+sum(sum(curb_value_bus_tot[:,start_time+1:start_time+T].*value.(U[:,:,3])))


# Dantzig-Wolfe algorithm on the same curb allocation Seattle Insignia problem

feas_sol=zeros(Int64,tot_curbs,T,E); # first trivial initial solution
for t=1:T
    for i=1:tot_curbs
        if i<=96
            feas_sol[i,t,1]=1;
        elseif i>96 && i<=192
            feas_sol[i,t,2]=1;
        elseif i>192 && i<=289
            feas_sol[i,t,3]=1;
        end
    end
end

feas_sol_2=zeros(Int64,tot_curbs,T,E); # second trivial initial solution
for t=1:T
    for i=1:tot_curbs
        if i<=96
            feas_sol_2[i,t,3]=1;
        elseif i>96 && i<=192
            feas_sol_2[i,t,1]=1;
        elseif i>192 && i<=289
            feas_sol_2[i,t,2]=1;
        end
    end
end

a=zeros(tot_curbs,T,E); # prices of curbes over T timesteps for E curb types
for i=1:tot_curbs
    for t=1:T
        a[i,t,1]=curb_value_paid_tot[i,start_time+t];
        a[i,t,2]=curb_value_cv_tot[i,start_time+t];
        a[i,t,3]=curb_value_bus_tot[i,start_time+t];
    end
end

#DW method
#start iterations
iter_length=30;   # number of iterations of DW
opt_sol=zeros(Float64,tot_curbs);
DW_sol=zeros(Float64,iter_length);  # DW optimal objective value
upd_sol=zeros(Float64,tot_curbs,T,E);
final_sol=zeros(Float64,tot_curbs,T,E);
iter_run_no=zeros(Int64,1);
iter_loc_no=zeros(Int64,1);
zsum=zeros(Float64,tot_curbs);
z_sol=zeros(Float64,tot_curbs,iter_length+1);
U_sol=zeros(Int64,tot_curbs,T,E,iter_length+2);  # allocation solution
U_sol[:,:,:,1].=feas_sol_2;#value.(U);  # initializing
U_sol[:,:,:,2].=feas_sol;#value.(U);
M_sol_time=zeros(Float64,iter_length);
S_sol_time=zeros(Float64,iter_length);
for iter_no=1:iter_length
    a1=zeros(Float64,tot_curbs,iter_no+1)
    z_local=zeros(Float64,tot_curbs,iter_no+1);
    for i=1:tot_curbs
        for j=1:iter_no+1
            a1[i,j]=sum(sum(a[i,h,g]*U_sol[i,h,g,j] for g=1:E) for h=1:T);
        end
    end
    start_M=time();
    JM_M=Model(with_optimizer(Gurobi.Optimizer, QCPDual=1, OutputFlag=0))
    # defining variables
    #@variable(JM,U[1:n,1:T,1:E]) # allocation variable
    @variable(JM_M,z[1:tot_curbs,1:iter_no+1])  # which solution to choose
    @variable(JM_M,x[1:T-1])  # change in allocation

    @objective(JM_M,Min,-sum(sum(a1.*z)))
    #@objective(JM,Min,-sum(sum(sum(a.*U_sol))))#+1*sum(sum(w)) # maximize linear revenue+ regularization for spacing
    # defining constraints

    @constraint(JM_M,con[i=1:tot_curbs], sum(z[i,:]) == 1);  # getting duals of sum(z) constraint
    @constraint(JM_M,z.>=0) # z is positive

    # getting duals of constraints on number of allocations and change in allocations between timesteps
    @constraint(JM_M,con1[t=1:T, e=1:E],sum(sum(U_sol[i,t,e,j].*z[i,j] for j=1:iter_no+1) for i=1:tot_curbs)<=t1max[e])
    @constraint(JM_M,con2[t=1:T, e=1:E],sum(sum(U_sol[i,t,e,j].*z[i,j] for j=1:iter_no+1) for i=1:tot_curbs)>=t1min[e])
    @constraint(JM_M,con3[t=2:T], sum(sum(sum((U_sol[i,t,e,j].*z[i,j]-U_sol[i,t-1,e,j].*z[i,j])^2 for j=1:iter_no+1) for i=1:tot_curbs) for e=1:E)<=2*b)


    status_M=optimize!(JM_M) # run the optimization
    M_sol_time[iter_no]=time()-start_M;
    # println("Optimal solution_master")
    # println(sum(sum(a1.*value.(z))))
    # println("Dual variable value")
    con_zsum=dual.(con);
    zsum .= dual.(con);
    if iter_no>=iter_length
        z_sol.=value.(z);
    end
    z_local.=value.(z);
    for i=1:tot_curbs
        for e=1:E
            for t=1:T
                upd_sol[i,t,e]=z_local[i,1:iter_no+1]'*U_sol[i,t,e,1:iter_no+1];
            end
        end
    end
    if iter_no>=iter_length
        final_sol.=upd_sol;
    end
    DW_sol[iter_no]=sum(sum(sum(a.*upd_sol)))   # optimal objective value from DW master problem
    # println(dual.(con))
    # println(con_zsum)
    # println("Primal variable value z")
    # println(value.(z))
    con_UL=dual.(con1);
    # println(dual.(con1))
    # println(con_UL)
    con_LL=dual.(con2);
    # println(dual.(con2))
    # println(con_LL)
    con_x=dual.(con3);
if iter_no<iter_length
    #start localized solution loop
    start_S=time();
    for loc_no=1:tot_curbs   # running sub-problems over each curb (in practice would be done in parallel to save time)
        #println(con_zsum[loc_no])
        #println(con_UL)
        #println(con_LL)
        JM_S=Model(with_optimizer(Gurobi.Optimizer, MIPGap=.0001, OutputFlag=0))
        # defining variables
        @variable(JM_S,U[1:T,1:E], Bin) # allocation variable
        #@variable(JM_S,x[1:T-1,1:E], Int) # change in allocation
        a2=a[loc_no,:,:];  # getting cost for particular curb
        #@variable(JM,w[1:T,1:E]) # regularization term for spacing
        # defining objective
        @objective(JM_S,Min,-sum(sum(a2.*U))-sum(sum(con_UL.*U))-sum(sum(con_LL.*U))-sum(sum(con_x.*U[2:T,:])))#+1*sum(sum(w)) # maximize linear revenue+ regularization for spacing
        # defining constraints
        for t=1:T
            @constraint(JM_S,sum(U[t,:])==1) # at a time and location only one allocation allowed
        end
        # for t=1:T-1
        #     for e=1:E
        #         @constraint(JM_S,x[t,e] .== U[t+1,e]-U[t,e])
        #     end
        # end
        # @constraint(JM_S,sum(sum(x.^2))<=b)

        status_S=optimize!(JM_S) # run the optimization
        # println("Local MIP optimal solution")
        # println(sum(sum(a2.*value.(U))))
        opt_sol[loc_no]=sum(sum(a2.*value.(U)));
        #println("check reduced cost")
        #println(-sum(sum(a2.*value.(U)))-sum(sum(con_UL.*value.(U)))-sum(sum(con_LL.*value.(U)))-sum(sum(con_x.*value.(U[2:T,:])))-con_zsum[loc_no])
        if iter_no<iter_length  # checking if a better optimal solution exists (if some add to feas solutions and rerun Master problem, otherwise discard)
            if -sum(sum(a2.*value.(U)))-sum(sum(con_UL.*value.(U)))-sum(sum(con_LL.*value.(U)))-sum(sum(con_x.*value.(U[2:T,:])))-con_zsum[loc_no] < 0
                U_sol[loc_no,:,:,iter_no+2]=round.(Int64,value.(U));
                iter_run_no[1]=iter_no;
                iter_loc_no[1]=loc_no;
                #println("reduced cost less than 0.......................")
            else
                U_sol[loc_no,:,:,iter_no+2]=U_sol[loc_no,:,:,iter_no+1]
            end
        end

    #end localized loops
    end
    S_sol_time[iter_no]=(time()-start_S)/tot_curbs;
#end iterations
end
end
for i=1:tot_curbs
    for e=1:E
        for t=1:T
            U_sol[i,t,e,iter_length+2]=round(Int64,z_sol[i,1:iter_length+1]'*U_sol[i,t,e,1:iter_length+1]);  # rounding out (hopefully negligible decimals after few iterations)
        end
    end
end

# for i=1:n
#     for j=1:iter_length+1
#         if z_sol[i,j]==1
#             U_sol[i,:,:,iter_length+2].=U_sol[i,:,:,j];
#         else
#             U_sol[i,:,:,iter_length+2].=0;
#         end
#     end
# end

println("DW_opt_sol")
println(sum(sum(sum(a.*U_sol[:,:,:,iter_length+2]))))
