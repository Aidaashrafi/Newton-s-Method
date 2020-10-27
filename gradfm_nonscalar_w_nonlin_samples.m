function gr = gradfm_nonscalar_w_nonlin_samples(x,y,n,neuron_array,xw_size)
    m = sum(neuron_array) - neuron_array(n,1);
    gr = zeros(xw_size,1);
    for i = 1 : size(y,1)
        gr(m+i,1) = x{1,n}(i) - y(i);
    end
end