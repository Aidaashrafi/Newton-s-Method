function lambda =  update_l_nonscalar_w_nonlin(p, lambda, neuron_array, n,ns)
b = 1 ;
while b < size(p,1) + 1
    for i = 1 : n 
        for j = 1 : ns*neuron_array(i)
            lambda{1,i}(j,1) = lambda{1,i}(j,1) + p(b);
            b = b + 1;
        end 
    end
end

