% ns = 113;
% x0 = [51;35];
% y = [0.9;0.6];
train_ind = importdata('train_ind.mat');
val_ind = importdata('val_ind.mat');
test_ind = importdata('test_ind.mat');
x=0:0.05:3*pi;
y=sin(x.^2);
x_train = x(train_ind);
x_val = x(val_ind);
x_test = x(test_ind);
y_train = y(train_ind);
y_val = y(val_ind);
y_test = y(test_ind);
x0 = x_train';
y = y_train';
x0_val = x_val';
y_val = y_val';
x0_test = x_test';
y_test = y_test';
ns = size(x0,1);
nt = size(x0_test,1);
nv = size(x0_val,1);

options.tol = 1e-4;
options.maxitr = 1000;
parameters.gamma = 0.9;
parameters.sigma = 0.3;
parameters.beta = 0.5;
k0 = 1;
n = 2;
rng(2);
%neuron_array = randi([2 4],n,1);
neuron_array = [6;1];

% x0 = rand(k0,1);
% y = rand(neuron_array(n,1),1);

% x0 = [51;35;14;2];
% y = 1;

x_c = cell(1,n); 
l_c = cell(1,n); 
for i = 1 : n
    x_c{1,i} = 1 * rand(ns*neuron_array(i,1),1);
    l_c{1,i} = rand(ns*neuron_array(i,1),1);
end

w_c = cell(1,n); 
w_c{1,1} = 1 * rand(neuron_array(1,1),k0);
for i = 2 : n
   w_c{1,i} = 1 * rand(neuron_array(i,1),neuron_array(i-1,1));
end

list_size_wx= [];
for i = 1 : size(x_c,2)
    list_size_wx = [list_size_wx ; size(x_c{1,i},1)];
end
for i = 1 : size(w_c,2)
    list_size_wx = [list_size_wx ; (size(w_c{1,i},1)*size(w_c{1,i},2))];
end
xw_size = sum(list_size_wx);

%%norm_data = y/sum(y);
%%norm_test = y_test/sum(y_test);
% norm_data = (y-min(y))/(max(y)-min(y));
% norm_test = (y_test-min(y_test))/(max(y_test)-min(y_test));
[x,w,l,w_min,I,I2,error_list_v,error_list_tr,error_list_t] = n_layer_solver_nonscalar_w_nonlin_samples(n, ns,nt, neuron_array, k0, x0, y, l_c, x_c, w_c, options, parameters,xw_size,nv,x0_val,y_val,x0_test,y_test);
x_pre_t = function_prediction(n,nt,w_min,x0_test);
error = immse(y_test , x_pre_t{1,n});

% % figure
% % plot(x0,y,x0,x{1,n},'--k')





        

    