function [x,w] = update_x_w_nonscalar_w_nonlin(x,w,neuron_array,n,t,p,xw_size,ns)
    a = 1;
    while a < (xw_size+1)
        for i = 1 : n
            for j = 1 : ns*neuron_array(i)
                x{1,i}(j,1) = x{1,i}(j,1) + (t*p(a));
                a = a + 1;
            end 
        end
        for i = 1 : n
            for j = 1 : size(w{1,i},1)
                for k = 1 : size(w{1,i},2)
                    w{1,i}(j,k) = w{1,i}(j,k) + (t*p(a));
                    a = a +1;
                end
            end
        end
    end
end
