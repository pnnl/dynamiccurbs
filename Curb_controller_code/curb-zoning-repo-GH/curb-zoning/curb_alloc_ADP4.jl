# Julia code for dynamic curb allocation using ADP4 method (refer to ITS paper)
# written by Nawaf (considring ADP-MonteCarlo+local cost function)
# curb allocation for Seattle (Insignia) without bus stop consideration (no multiple curb allocations)
nb=26; # no of blocks
T=10; # no of timesteps (optimization horizon)
E=3; # no of curb types (1=paid parking, 2=commercial vehicles, 3=busus/public)

using CSV
using DataFrames

# read the curb length data
curb_data=CSV.read("./space_val_data/Iter2/ekey_length.csv", DataFrame)  # reading node data
tot_curbs=sum(curb_data[1:nb,3]); # total number of curbs (289 for Seattle Insignia)
loc=3; # no of demand locations
tot_T=96; # total time horizon of data available

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

# finish reading data

#starting ADP code
switch_allow=100; # no of allowed allocation changes in curb zoning between two time-steps
 b=8; # no of changes in a particular curb summed over the timesteps
 # tmin=[260; 260; 260]; # minimum allocations over time
 # tmax=[1500; 1500; 1500]; # maximum allocations over time
t1min=[50; 50; 50]; # minimum allocations at one timestep (of a type of allocation)
t1max=[125; 125; 125]; # maximum allocations at one timestep

# optimization problem starts
MC_scen=100; # how many MC loops to run in curb allocation per iteration (innerloop)
start_time=49; # start the optimization horizon at this timestep (from tot_T=96)
w=zeros(MC_scen+1,T); # storing value of regularization term of objective function over MC_scen
V=zeros(MC_scen+1,T); # storing value of total objective function over MC_scen
Vmin=zeros(T,1);  # maximum objective amongst MC_scen at each time step
U=zeros(Int64,tot_curbs,T); # allocation variable
U_S=zeros(Int64,MC_scen+1,tot_curbs,T); # storing allocations over MC_scen random allocations
U_paid=zeros(Int64,tot_curbs,T); #whether paid parking or not
U_cv=zeros(Int64,tot_curbs,T); #whether commercial or not
U_bus=zeros(Int64,tot_curbs,T); #whether bus-stop or not
x=zeros(Float64,tot_curbs); # change in allocation (from one time step to the next)
y=zeros(Float64,tot_curbs); # sum of allocation change (need to limit this)
U_pr=zeros(Int64,MC_scen+1,tot_curbs,T); # storing values of allocation in previous iterations
V_pr=zeros(Float64,MC_scen+1,T); # storing value of objective function in previous iterations
a_rand=rand((1,2,3),tot_curbs); # random curb allocations over tot_curbs (1=pp, 2=cv, 3=bus)

for t_step=1:T
    U[:,t_step].=a_rand; # initializing curb allocation to random allocation (a_rand)
end

#(1=paid parking, 2=commercial vehicles, 3=busus/public)
#w=zeros(Float64,T) # regularization term for spacing
ITER=40; # no of MonteCarloC iterations (outerloop)
obj_value=zeros(Float64,ITER); # storing the objective value at each MC iteration
sol_time=zeros(Float64,ITER);
# defining objective
for iter_loop=1:ITER # looping over MC iterations
    start=time();
    # U_paid=zeros(Int64,tot_curbs,T); #whether curb is paid parking or not (0 or 1)
    # U_cv=zeros(Int64,tot_curbs,T); #whether commercial or not
    # U_bus=zeros(Int64,tot_curbs,T); #whether bus-stop or not

    for t_step=1:T   # intializing values of U_paid, U_cv and U_bus based on initial allocation of U
        for v=1:tot_curbs
            if U[v,t_step]==1
                U_paid[v,t_step]=1;
            elseif U[v,t_step]==2
                U_cv[v,t_step]=1;
            elseif U[v,t_step]==3
                U_bus[v,t_step]=1;
            end
        end
    end

    for t_step=1:T  # looping over time steps (DP mechanism)
        for i=1:tot_curbs
            U_S[1,i,t_step]=U[i,t_step];  # initializing the first MC_scen loop in U_S
        end
        V[1,t_step]=Vmin[t_step]; # intializing the first MC_scen loop in V
        for rand_change=1:MC_scen  # make 100 random changes (100 times)
            if mod(iter_loop,10)==0 && rand_change==1 # every 10th iteration start with a completely random allocation again (exploitation vs exploration)
                b_rand=rand((1,2,3),tot_curbs); # new random allocation
                U_S[rand_change+1,:,t_step]=b_rand'; # storing random allocation in U_S
                for t_curb=1:tot_curbs    # looping to check which ones are pp, cv or bus
                    if U_S[rand_change+1,t_curb,t_step]==1
                        U_paid[t_curb,t_step]=1;
                        U_cv[t_curb,t_step]=0;
                        U_bus[t_curb,t_step]=0;
                    elseif U_S[rand_change+1,t_curb,t_step]==2
                        U_cv[t_curb,t_step]=1;
                        U_paid[t_curb,t_step]=0;
                        U_bus[t_curb,t_step]=0;
                    elseif U_S[rand_change+1,t_curb,t_step]==3
                        U_bus[t_curb,t_step]=1;
                        U_paid[t_curb,t_step]=0;
                        U_cv[t_curb,t_step]=0;
                    end
                end
            else     # not 10th iteration or not first rand_change in MC_scen loop
                for i=1:tot_curbs
                    U_S[rand_change+1,i,t_step]=U_S[1,i,t_step]; # intialize to first allocation
                end
                for t_curb=1:tot_curbs   # looping to check which ones are pp, cv or bus
                    if U_S[rand_change+1,t_curb,t_step]==1
                        U_paid[t_curb,t_step]=1;
                        U_cv[t_curb,t_step]=0;
                        U_bus[t_curb,t_step]=0;
                    elseif U_S[rand_change+1,t_curb,t_step]==2
                        U_cv[t_curb,t_step]=1;
                        U_paid[t_curb,t_step]=0;
                        U_bus[t_curb,t_step]=0;
                    elseif U_S[rand_change+1,t_curb,t_step]==3
                        U_bus[t_curb,t_step]=1;
                        U_paid[t_curb,t_step]=0;
                        U_cv[t_curb,t_step]=0;
                    end
                end
                curb_change=rand(1:tot_curbs,10); # randomly choose 10 locations amongst tot_curbs to modify according to local costs
                for v=1:length(curb_change)
                    # sort chosen curbs according to value of each allocation type
                    c1=sortperm([curb_value_paid_tot[curb_change[v],start_time+t_step];curb_value_cv_tot[curb_change[v],start_time+t_step];curb_value_bus_tot[curb_change[v],start_time+t_step]])
                    U_S[rand_change+1,curb_change[v],t_step]=convert(Int64,c1[3]); # change allocation based on local costs
                    if c1[3]==1
                        U_paid[curb_change[v],t_step]=1;
                        U_cv[curb_change[v],t_step]=0;
                        U_bus[curb_change[v],t_step]=0;
                    elseif c1[3]==2
                        U_cv[curb_change[v],t_step]=1;
                        U_paid[curb_change[v],t_step]=0;
                        U_bus[curb_change[v],t_step]=0;
                    elseif c1[3]==3
                        U_bus[curb_change[v],t_step]=1;
                        U_cv[curb_change[v],t_step]=0;
                        U_paid[curb_change[v],t_step]=0;
                    end
                end
            end
            #w[rand_change+1,t_step]=.01*((U_paid[:,t_step]'*dist_matrix*U_paid[:,t_step])+(U_cv[:,t_step]'*dist_matrix*U_cv[:,t_step])+(U_bus[:,t_step]'*dist_matrix*U_bus[:,t_step]));
            w[rand_change+1,t_step]=0; # ignore curb distance for now
            # calculate objective function based on generated curb allocation
            V[rand_change+1,t_step]=w[rand_change+1,t_step]-U_paid[:,t_step]'*curb_value_paid_tot[:,start_time+t_step]-U_cv[:,t_step]'*curb_value_cv_tot[:,start_time+t_step]-U_bus[:,t_step]'*curb_value_bus_tot[:,start_time+t_step];
            # check to see if constraints on curb allocation are satisfied, if not objective is infinity (to discard that allocation)
            if sum(U_paid[:,t_step])<t1min[1] || sum(U_paid[:,t_step])>t1max[1]
                V[rand_change+1,t_step]=10^8;
            elseif sum(U_cv[:,t_step])<t1min[2] || sum(U_cv[:,t_step])>t1max[2]
                V[rand_change+1,t_step]=10^8;
            elseif sum(U_bus[:,t_step])<t1min[3] || sum(U_bus[:,t_step])>t1max[3]
                V[rand_change+1,t_step]=10^8;
            end
        end

        c=sortperm(V[:,t_step]) # sort all MC_scen allocations according to objective function value
        for i=1:tot_curbs
            U[i,t_step]=U_S[c[1],i,t_step]; # allocate best performing allocation amongst MC_scen to U
        end
        Vmin[t_step]=V[c[1],t_step]; # allocate best objective function value to Vmin
        #println("Vmin1")
        #println(Vmin[t_step])
        # checking if the curb allocation change exceeds limits
        d=1; # check flag to see if curb allocation satisfies inter-step change limits
        test_no=1; # if not satisfied looping over the less optimal allocations till a satisfactory one is found
        while d==1 && test_no<=MC_scen+1 && t_step!=1
            for e=1:MC_scen+1
                x .= U_S[c[e],:,t_step]-U_pr[test_no,:,t_step-1];  # finding change in curb allocation with the previous timestep
                for f=1:tot_curbs
                    if x[f]!=0   # if change detected add 1 to y
                        y[f]=1;
                    else
                        y[f]=0;
                    end
                end
                if sum(y)<switch_allow && V[c[e],t_step]<0 # checking if constraint on change limit is satisfied
                    d=0;  # raising the flag
                    for i=1:tot_curbs
                        U[i,t_step]=U_S[c[e],i,t_step];         # saving this allocations
                        U[i,t_step-1]=U_pr[test_no,i,t_step-1]; # also saving previous timestep
                    end
                    Vmin[t_step]=V[c[e],t_step]; # saving the objective function
                    #println(test_no)
                    #println(e)
                    #println("Vmin")
                    #println(Vmin[t_step])
                    Vmin[t_step-1]=V_pr[test_no,t_step-1];
                    #println("Vmin_prev")
                    #println(Vmin[t_step-1])
                    break
                end
            end
            test_no=test_no+1;  # moving to next allocation over MC_scen
        end
        for g=1:length(c)
            U_pr[g,:,t_step].=U_S[c[g],:,t_step]; # updating previous timestep values
            V_pr[g,t_step]=V[c[g],t_step];
        end
        U_pr[1,:,t_step].=U[:,t_step]
        V_pr[1,t_step]=Vmin[t_step]
        if d!=0 && t_step!=1   # if no satisfactory allocation found in the list MC_scen
            #println("using previous time")
            U[:,t_step].=U[:,t_step-1];  # keeping the previous time-step allocation
            Vmin[t_step]=Vmin[t_step-1];
            U_pr[1,:,t_step].=U[:,t_step];
            V_pr[1,t_step]=Vmin[t_step];
        end
    end
    obj_value[iter_loop]=sum(Vmin);  # objective value found by summing over the time-steps
    sol_time[iter_loop]=time()-start;
end
#qtime=time()-start; # total run-time
# for i=1:289
#     if U[i,2] ==1
#         println("paid")
#     elseif U[i,2] ==2
#         println("cv")
#     else
#         println("bus")
#     end
# end
