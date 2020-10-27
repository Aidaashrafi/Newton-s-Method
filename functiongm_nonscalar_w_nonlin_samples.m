function g = functiongm_nonscalar_w_nonlin_samples(n,ns,x,w,x0)
    obj_fg = nonlinfunction();
    obj_lin_fg = linfunction();
%     for p = 1 : ns
%         g{1,1} = x{1,1}(((p-1)*size(x{1,1},1))+1:(p*size(x{1,1},1))) - (obj_fg.value(w{1,1}*x0(((p-1)*size(x0,1))+1:(p*size(x0,1))))) ;
%     end
    g{1,1} = x{1,1} - obj_fg.value(kron(eye(ns),w{1,1})*x0);
%     for s = 1 : ns
%         for i = 2 : n
%             g{i,1} = x{1,i}(((s-1)*size(x{1,i},1))+1:(s*size(x{1,i},1))) - (obj_fg.value(w{1,i} * x{1,i-1}(((s-1)*size(x{1,i-1},1))+1:(s*size(x{1,i-1},1)))));
%         end
%     end
    if n > 2
        for i = 2 : n-1
            g{i,1} = x{1,i} - obj_fg.value(kron(eye(ns),w{1,i})*x{1,i-1});
        end
        g{n,1} = x{1,n} - obj_lin_fg.value(kron(eye(ns),w{1,n})*x{1,n-1});
    else
        g{n,1} = x{1,n} - obj_lin_fg.value(kron(eye(ns),w{1,n})*x{1,n-1}); 
    end
end