%script to determine A and B matrices from SeaTac data using the model
%formulation \dot(x)=Ax+Bu; x=[DF;DS;AF;AS], u=[TA;TD];
clear all
full_data=readtable('seatac_vms1_vms2_treatment_ctl','Range','A1:K11914'); % reading entire dataset
loc_wo_con=full_data.control2==1; % finding location of rows where control2==1 (no control input, u=0)
data_wo_con=full_data(loc_wo_con,:); % data with control2==1 (no control input, u=0)

% finding A matrix using data_wo_con

x_k=data_wo_con(1:end-1,2:5); % state at time-step k
x_k1=data_wo_con(2:end,2:5); % state at time-step k+1
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
A1=x_k1'*pinv(x_k');  % least square estimate

% finding B matrix by setting A=A1 and (treat_arr>0 or treat_dep>0)

loc_with_con=full_data.treat_arr>0 | full_data.treat_dep>0; % finding location of rows where treat>0 (u~=0)
data_with_con=full_data(loc_with_con,:); % data with control2~=1 (u~=0)

x_k=data_with_con(1:end-1,2:5); % state at time-step k
x_k1=data_with_con(2:end,2:5); % state at time-step k+1
u_k=data_with_con(1:end-1,10:11); % input at time-step k (TA and TD)
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
u_k=table2array(u_k); % convert to a matrix

B1=(x_k1'-A1*(x_k'))*pinv(u_k');

% finding A and B simultaneously

x_k=full_data(1:end-1,2:5); % state at time-step k
x_k1=full_data(2:end,2:5); % state at time-step k+1
u_k=full_data(1:end-1,10:11); % input at time-step k (TA and TD)
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
u_k=table2array(u_k); % convert to a matrix

phi_k=[x_k u_k];  % concatenate x_k and u_k
A_B=x_k1'*pinv(phi_k');
A2=A_B(:,1:4);
B2=A_B(:,5:6);

% obtaining time of day A matrices (96 matrices for each 15 min time slot)


full_data_t=table2array(full_data(:,2:end));
A3=zeros(4,4,96);
for i=1:96
    x_k=zeros(124,4);
    x_k1=zeros(124,4);
    for j=1:124
        x_k(j,:)=full_data_t(96*(j-1)+i,1:4);
        x_k1(j,:)=full_data_t(96*(j-1)+i+1,1:4);
    end
    A3(:,:,i)=x_k1'*pinv(x_k');
end

%using new formulation \dot(x)=Ax+Bu; x=[DF;DS;AF;AS;DF_dot;DS_dot;AF_dot;AS_dot], u=[TA;TD];

% finding new A matrix using data_wo_con

x_k=data_wo_con(1:end-1,2:5); % state at time-step k
x_k1=data_wo_con(2:end,2:5); % state at time-step k+1
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
x_dot_k1=x_k1-x_k; % first order change in states time-step k+1
x_dot_k=[zeros(1,4);x_dot_k1(1:end-1,:)]; % first order change in states time-step k
x_k_total=[x_k x_dot_k]; % complete state at time-step k
x_k1_total=[x_k1 x_dot_k1]; % complete state at time-step k+1
A1_total=x_k1_total'*pinv(x_k_total');  % least square estimate

% finding B matrix by setting A=A1_total and (treat_arr>0 or treat_dep>0)

x_k=data_with_con(1:end-1,2:5); % state at time-step k
x_k1=data_with_con(2:end,2:5); % state at time-step k+1
u_k=data_with_con(1:end-1,10:11); % input at time-step k (TA and TD)
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
x_dot_k1=x_k1-x_k; % first order change in states time-step k+1
x_dot_k=[zeros(1,4);x_dot_k1(1:end-1,:)]; % first order change in states time-step k
x_k_total=[x_k x_dot_k]; % complete state at time-step k
x_k1_total=[x_k1 x_dot_k1]; % complete state at time-step k+1
u_k=table2array(u_k); % convert to a matrix

B1_total=(x_k1_total'-A1_total*(x_k_total'))*pinv(u_k');

% finding new A and B simultaneously

x_k=full_data(1:end-1,2:5); % state at time-step k
x_k1=full_data(2:end,2:5); % state at time-step k+1
u_k=full_data(1:end-1,10:11); % input at time-step k (TA and TD)
x_k=table2array(x_k); % convert to a matrix
x_k1=table2array(x_k1); % convert to a matrix
u_k=table2array(u_k); % convert to a matrix
x_dot_k1=x_k1-x_k; % first order change in states time-step k+1
x_dot_k=[zeros(1,4);x_dot_k1(1:end-1,:)]; % first order change in states time-step k
x_k_total=[x_k x_dot_k]; % complete state at time-step k
x_k1_total=[x_k1 x_dot_k1]; % complete state at time-step k+1

phi_k_total=[x_k_total u_k];  % concatenate x_k and u_k
A_B_total=x_k1_total'*pinv(phi_k_total');
A2_total=A_B_total(:,1:8);
B2_total=A_B_total(:,9:10);
