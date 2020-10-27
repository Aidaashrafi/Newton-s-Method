function value =  functionfm_nonscalar_w_nonlin_samples(x,y,n)
    value = 0;
    for i = 1 : size(y,1)
    %n = size(x,2);
        value = value + ((1.0/2)*((norm(x{1,n}(i) - y(i)))^2));
    end
end