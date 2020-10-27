function [x,w,t] = linesearch_nonscalar_w_nonlin_sampless(x, x0, w, y, m, n, Jpk, pk, gamma, beta,rho,r,k,jac,l,xw_size,rr)
    x0_int = x0;
    y_int = y;
    x_int = x;
    w_int = w;
    l_int = l;
    
	% Line search using Armijo conditions and backtracking
	% fun - scalar function
	% x0 - initial guess
	% Jpk - J is gradient of fun at x0, Jpk is J'*pk
	% pk - search direction
	% gamma - armijo condition scaling function
	% beta - backtracking parameter

	% make sure gamma and beta are in reasonable range
	if gamma <= 0 || 1 <= gamma
		error('gamma must be in (0,1)')
	end

	if beta <= 0 || 1 <= beta
		error('beta must be in (0,1)')
    end
    
    %%****************************************
    %% number of iterations
	too_many_steps_counter = 100;
    
    %%****************************************
	%% initialize t
	t = 1;
    ff = functionfm_nonscalar_w_nonlin_samples(x,y,n);
    fg = functiongm_nonscalar_w_nonlin_samples(n,ns,x,w,x0);
    vec_fg = make_vector(fg);
        
    %%****************************************
    %% merit function: phi(x) =  f(x) + (rho * norm(g,1))
    f0 = merit_funm_nonscalar_w_nonlin(ff,vec_fg,rho);
    
    %%****************************************
    %% updating the x,w with initial p
    %     [x,w] = update_x_w(x,w,m,n,t,pk);
    x_old = x;
    w_old = w; 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    count = 0;
    % %     v1 = f0
    % %     v2 = gamma*t*Jpk
    % %     v1+v2
    % %     v3 = merit_funm_nonscalar_w(functionfm(x,y,m,n),make_vector(functiongm_nonscalar_w(n,x,w,x0)),rho)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%****************************************
    %% Armijo condition: phi(x+(t*pk)) = phi(x) + gamma * t * Jpk
    %% phi(x+(t*pk)) = merit_funm_nonscalar_w(functionfm(x,y,m,n),make_vector(functiongm_nonscalar_w(n,x,w,x0)),rho)
    %% phi(x) = f0, Jpk = J'*pk - rho*norm(g,1), J = gradient f(x)

	while count==0 | (merit_funm_nonscalar_w_nonlin(functionfm_nonscalar_w_nonlin_samples(x,y,n),make_vector(functiongm_nonscalar_w_nonlin_samples(n,ns,x,w,x0)),rho) > f0 + (gamma*t*Jpk) ) 
    %while 1
        count = count + 1;
        if 1
        %if merit_funm_nonscalar_w(functionfm(x,y,m,n),make_vector(functiongm_nonscalar_w(n,x,w,x0)),rho) > f0 + (gamma*t*Jpk)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %mm = merit_funm_nonscalar_w(functionfm(x,y,m,n),make_vector(functiongm_nonscalar_w(n,x,w,x0)),rho)
            %mf = f0 + (gamma*t*Jpk)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%****************************************
            %% cahnge the step size and update x,w
            t = beta*t   ;
%             t = 0; 
            [x,w] = update_x_w_nonscalar_w_nonlin(x_old,w_old,m,n,t,pk,xw_size);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %mm2 = merit_funm_nonscalar_w(functionfm(x,y,m,n),make_vector(functiongm_nonscalar_w(n,x,w,x0)),rho)
            %mf2 = f0 + (gamma*t*Jpk)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end %end for if
                    %%****************************************
        %% if the line search takes too many iterations, throw an error
        too_many_steps_counter = too_many_steps_counter - 1;
        if too_many_steps_counter == 0
            formatSpec = 'line search fail - took too many iterations, norm(r) is %f and iteration is %d\n';
            fprintf(formatSpec,r,k)
            fprintf('jacobian is\n')
            jac
%             formatSpec2 = 'x0 is %f and y is %f';
%             fprintf(formatSpec2,x0_int,y_int)
%             fprintf('x is\n')
%             x
%             fprintf('w is\n')
%             w
%             fprintf('l is\n')
%             l
            formatSpec2 = 'x0 is %f, x1 is %f, x2 is %f, w1 is %f, w2 is %f, l1 is %f, l2 is %f and y is %f\n';
            fprintf(formatSpec2,x0_int,x_int(1,1),x_int(2,1),w_int(1,1),w_int(2,1),l_int(1,1),l_int(2,1),y_int)
            error('line search fail - took too many iterations')
            %break
        end
    end
%     [x,w] = update_x_w(x_old,w,m,n,t,pk);
    count
    formatSpec = 'norme(r) is %f and iteration is %d\n';
    fprintf(formatSpec,r,k)
    fprintf('r is\n')
    rr
    fprintf('jacobian is\n')
    jac
    fprintf('*******************************\n')
end