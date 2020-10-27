function r_c = calculate_r_nonscalar_w_nonlin_samples(x0,y,w,x,l,n,ns,neuron_array,k0)
    r_c = cell(3*n,1);
    obj1 = nonlinfunction();
    obj2 = linfunction();
    if n > 2 
        for i = 1 : (n - 2)  
            r_c{i,1} = l{1,i} - ((kron(eye(ns),w{1,i+1}')) * diag(obj1.grad((kron(eye(ns),w{1,i+1}))*x{1,i}))*l{1,i+1});
        end
        r_c{n-1,1} = l{1,n-1} - ((kron(eye(ns),w{1,n}')) * diag(obj2.grad((kron(eye(ns),w{1,n}))*x{1,n-1}))*l{1,n});
        %size_neu_arr = size(neuron_array,1);
        r_c{n,1} = x{1,n} - y + l{1,n};

        r_c{n+1,1} = zeros(neuron_array(1)*k0,1);
        for p = 1 : ns
            r_c{n+1,1} = r_c{n+1,1} + (-kron((diag(obj1.grad(w{1,1}*x0(((p-1)*k0)+1:(p*k0))))*l{1,1}(((p-1)*neuron_array(1))+1:(p*neuron_array(1)))),x0(((p-1)*k0)+1:(p*k0))));
        end

        b = n+2;
        for i = 2 : n-1
            r_c{b,1} = zeros(neuron_array(i)*neuron_array(i-1),1);
            for s = 1 : ns
                r_c{b,1} = r_c{b,1} + (-kron((diag(obj1.grad(w{1,i}*x{1,i-1}(((s-1)*neuron_array(i-1))+1:(s*neuron_array(i-1)))))*l{1,i}(((s-1)*neuron_array(i))+1:(s*neuron_array(i)))),x{1,i-1}(((s-1)*neuron_array(i-1))+1:(s*neuron_array(i-1))))); 
            end
            b = b+1;
        end
        r_c{b,1} = zeros(neuron_array(n)*neuron_array(n-1),1);
        for s = 1 : ns
            r_c{b,1} = r_c{b,1} + (-kron((diag(obj2.grad(w{1,n}*x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)))))*l{1,n}(((s-1)*neuron_array(n))+1:(s*neuron_array(n)))),x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))))); 
        end
        b = b+1;
        r_c{b,1} = x{1,1} - (obj1.value(kron(eye(ns),w{1,1}) * x0));
 
        b = b+1;
        for i = 2 : n-1
            r_c{b,1} = x{1,i} - (obj1.value(kron(eye(ns),w{1,i}) * x{1,i-1}));
            b = b+1;
        end
        r_c{b,1} = x{1,n} - (obj2.value(kron(eye(ns),w{1,n}) * x{1,n-1}));
    else
        r_c{n-1,1} = l{1,n-1} - ((kron(eye(ns),w{1,n}')) * diag(obj2.grad((kron(eye(ns),w{1,n}))*x{1,n-1}))*l{1,n});
        %size_neu_arr = size(neuron_array,1);
        r_c{n,1} = x{1,n} - y + l{1,n};

        r_c{n+1,1} = zeros(neuron_array(1)*k0,1);
        for p = 1 : ns
            r_c{n+1,1} = r_c{n+1,1} + (-kron((diag(obj1.grad(w{1,1}*x0(((p-1)*k0)+1:(p*k0))))*l{1,1}(((p-1)*neuron_array(1))+1:(p*neuron_array(1)))),x0(((p-1)*k0)+1:(p*k0))));
        end

        b = n+2;
        r_c{b,1} = zeros(neuron_array(n)*neuron_array(n-1),1);
        for s = 1 : ns
            r_c{b,1} = r_c{b,1} + (-kron((diag(obj2.grad(w{1,n}*x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)))))*l{1,n}(((s-1)*neuron_array(n))+1:(s*neuron_array(n)))),x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))))); 
        end

        r_c{b+1,1} = x{1,1} - (obj1.value(kron(eye(ns),w{1,1}) * x0));
        b = b+1;
        r_c{b+1,1} = x{1,n} - (obj2.value(kron(eye(ns),w{1,n}) * x{1,n-1}));
    end
end
    