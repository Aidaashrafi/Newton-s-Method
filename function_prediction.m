function x = function_prediction(n,nt,w,x0_t)
    obj_fp = nonlinfunction();
    obj_fp2 = linfunction();
%     for p = 1 : ns
%         g{1,1} = x{1,1}(((p-1)*size(x{1,1},1))+1:(p*size(x{1,1},1))) - (obj_fg.value(w{1,1}*x0(((p-1)*size(x0,1))+1:(p*size(x0,1))))) ;
%     end
    x{1,1} = obj_fp.value(kron(eye(nt),w{1,1})*x0_t);
%     for s = 1 : ns
%         for i = 2 : n
%             g{i,1} = x{1,i}(((s-1)*size(x{1,i},1))+1:(s*size(x{1,i},1))) - (obj_fg.value(w{1,i} * x{1,i-1}(((s-1)*size(x{1,i-1},1))+1:(s*size(x{1,i-1},1)))));
%         end
%     end
    for i = 2 : n-1
        x{1,i} = obj_fp.value(kron(eye(nt),w{1,i})*x{1,i-1});
    end
    x{1,n} = obj_fp2.value(kron(eye(nt),w{1,n})*x{1,n-1});
end