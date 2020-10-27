function [x,w,l,w_min,I,I2,error_list_v,error_list_tr,error_list_t] = n_layer_solver_nonscalar_w_nonlin_samples(n, ns, nt, neuron_array, k0, x0, y, l, x, w, options, parameters,xw_size,nv,x0_val,y_val,x0_test,y_test)

    sigma = parameters.sigma;
    error_list_v = [];
    error_list_tr = [];
    error_list_t = [];
    w_list = [];
    list_r = [];
    list_k = [];
    list_itr = [];
    list_abs_error = [];
    for k = 1 : options.maxitr
        %%****************************************
        %% calculating the gradient of langrangian with respect to x,w,l
        r = calculate_r_nonscalar_w_nonlin_samples(x0,y,w,x,l,n,ns,neuron_array,k0);
        r = cell2mat(r);
        size_r = size(r,1);
        r1 = norm(r);
        list_r = [list_r;r1];
        list_k = [list_k;k];
        abs_error = norm(x{1,n}-y);
        list_abs_error = [list_abs_error;abs_error];
        %%****************************************
        %% minimizing the gradient
        if r1 < options.tol
            break;
        end

        %%****************************************
        %% calculating the jacobian of the gradient
        jac = calculate_jac_nonscalar_w_nonlin_samples(x0,w,x,l,n,ns,neuron_array,k0);
        jac = cell2mat(jac);


        % regularized Newton: based on Ueda Yamahita 2010 
        c1 = 1; c2= 0.01; delta= 0.5;
        lam_k = max(0,-real(eigs(jac,1,'smallestabs'))); 
        mu_k = c1*lam_k + c2* r1^delta; 
%         mu_k = 0;    
        p = -((jac+mu_k*eye(size(r,1)))\r);
        %%****************************************
        %% update amounts for x and w
        pk = p(1 : xw_size);
        
        %%****************************************
        %% update amounts for l
        a = p(xw_size + 1 : end);
        
        
        mat2cell(pk,size(pk,1),1);
        mat2cell(a,size(a,1),1);
        
        %%****************************************
        %% updating lambdas(l)
        l = update_l_nonscalar_w_nonlin(a, l, neuron_array, n,ns);
        
        %%****************************************
        %% Armijo constant
        l_vec = [];
        for i = 1 : n
            l_vec = [l_vec;l{1,i}];
        end
        %% rho is 1/Mu
        rho = norm(l_vec,inf) + sigma;
    
        gamma = parameters.gamma;
        
        %%****************************************
        %% multiplying t by this factor in each iteration
	    beta = parameters.beta ;
        
        %%****************************************
        %% computing gradient of f(x)
        %% f(x) = 1/2|x(n)-y|^2
        J = gradfm_nonscalar_w_nonlin_samples(x,y,n,neuron_array,xw_size);
        
        %%****************************************
        %% computing g(x)
        %% [x(1) - w(1)*x(0); x(2) - w(2)*x(1); ...]
        g = functiongm_nonscalar_w_nonlin_samples(n,ns,x,w,x0);
        g_vec = [];
        for i = 1 : size(g,1)
            g_vec = [g_vec;g{i,1}];
        end
        %%****************************************
        %% finding the optimal step iteratively
        [x,w,t] = linesearch(x, x0, w, y, neuron_array, n, ns, J'*pk - rho*norm(g_vec,1), pk, gamma, beta,rho,r1,k,jac,l,xw_size,r);
        x_pre_v = function_prediction(n,nv,w,x0_val);
%         error_v = y_val - x_pre_v{1,n};
%         sum_err = sumabs(error_v);
        sum_err = immse(y_val , x_pre_v{1,n});
        error_list_v = [error_list_v ; sum_err];
        w_list = [w_list ; w];
        x_pre_tr = function_prediction(n,ns,w,x0);
%         error_tr = y - x_pre_tr{1,n};
%         sum_err_tr = sumabs(error_tr);
        sum_err_tr = immse(y , x_pre_tr{1,n});
        error_list_tr = [error_list_tr ; sum_err_tr];
        
        x_pre_t = function_prediction(n,nt,w,x0_test);
        sum_err_t = immse(y_test , x_pre_t{1,n});
        error_list_t = [error_list_t ; sum_err_t];
        list_itr = [list_itr ; k];

    end
    [M,I] = min(error_list_v);
    [M2,I2] = min(error_list_tr);
    w_min = w_list(I,1:n);
%     end
    figure
    pl = plot(list_itr,error_list_tr,'b',list_itr,error_list_v,'g',list_itr,error_list_t,'r')
    set(pl,'LineWidth',2)
    legend('trainset','validationset','testset')

%     hold on 
%     semilogy(list_itr,error_list_t)
    xlabel('iteration')
    ylabel('mse')

    
    figure
    sm = semilogy(list_k,list_r)
    set(sm,'LineWidth',2)
    xlabel('iteration')
    ylabel('||r||')
%     figure
%     semilogy(list_k,list_abs_error)
%     xlabel('iteration')
%     ylabel('||xn-y||')
%    x
%    w
%    l
end
