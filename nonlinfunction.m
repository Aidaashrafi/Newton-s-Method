function obj = nonlinfunction()
    obj.value = @(x) compute_value(x);
    obj.grad = @(x) compute_grad(x);
    obj.grad2 = @(x) compute_grad2(x);
end
function vec_value = compute_value(x)
    vec_value = [];
    for i = 1 : size(x)
        fac = 0.001;
        expo_neg = exp(- x(i));
        plusone = expo_neg + 1;
        value = 1.0 / plusone;
        vec_value = [vec_value ; value];
    end
end
function vec_grad = compute_grad(x)
    vec_grad = [];
    for i = 1 : size(x)
        val = compute_value(x(i));
        grad = (1-val)*(val);
        vec_grad = [vec_grad ; grad];
    end
end
function vec_grad2 = compute_grad2(x)
    vec_grad2 = [];
    for i = 1 : size(x)
        val2 = compute_value(x(i));
        grad2 = (((1-val2)*(val2))-(2*(val2^2)*(1-val2)));
        vec_grad2 = [vec_grad2 ; grad2];
    end
end