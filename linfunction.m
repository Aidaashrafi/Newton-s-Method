function obj = linfunction()
    obj.value = @(x) compute_value(x);
    obj.grad = @(x) compute_grad(x);
    obj.grad2 = @(x) compute_grad2(x);
end
function vec_value = compute_value(x)
    vec_value = x;
end
function vec_grad = compute_grad(x)
    vec_grad = ones(size(x,1),1);
end
function vec_grad2 = compute_grad2(x)
    vec_grad2 = zeros(size(x,1),1);
end