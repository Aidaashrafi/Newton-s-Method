function jac = calculate_jac_nonscalar_w_nonlin_samples(x0,w,x,l,n,ns,neuron_array,k0)
    obj_jac = nonlinfunction();
    obj_jac2 = linfunction();
    jac = cell((3*n),(3*n));
    list_size_vars= [];
    for i = 1 : size(x,2)
        list_size_vars = [list_size_vars ; size(x{1,i},1)];
    end
    for i = 1 : size(w,2)
        list_size_vars = [list_size_vars ; (size(w{1,i},1)*size(w{1,i},2))];
    end
    for i = 1 : size(l,2)
        list_size_vars = [list_size_vars ; size(l{1,i},1)];
    end
    for i = 1 : 3*n
        for j = 1 : 3*n
            jac{i,j} = zeros((list_size_vars(i,1)),(list_size_vars(j,1)));
        end
    end

%     for i = 1 : n
%         for j = 1 : 3*n
%             jac{j,n+i} = zeros(size(jac{j,n+i},1),size(jac{j,n+i},2)/ns);
%         end 
%     end
    if n > 2
        for i = 1 : (n - 2)
            for s = 1 : ns
                for j = 1 : neuron_array(i+1)
                    jac{i,i}(((s-1)*neuron_array(i))+1:(s*neuron_array(i)),((s-1)*neuron_array(i))+1:(s*neuron_array(i))) = jac{i,i}(((s-1)*neuron_array(i))+1:(s*neuron_array(i)),((s-1)*neuron_array(i))+1:(s*neuron_array(i))) - (l{1,i+1}(((s-1)*neuron_array(i+1))+j) * obj_jac.grad2(w{1,i+1}(j,:)*x{1,i}(((s-1)*neuron_array(i))+1:(s*neuron_array(i)))) * diag(w{1,i+1}(j,:)') * kron(ones(neuron_array(i),1),w{1,i+1}(j,:)));
                end
            end
            for ss = 1 : ns
                my_arr = [];
                for a = 1 : neuron_array(i+1)
                    my_arr = [my_arr -(l{1,i+1}(((ss-1)*neuron_array(i+1))+a) * ((obj_jac.grad(w{1,i+1}(a,:)*x{1,i}(((ss-1)*neuron_array(i))+1:(ss*neuron_array(i)))) * eye(neuron_array(i))) + (obj_jac.grad2(w{1,i+1}(a,:)*x{1,i}(((ss-1)*neuron_array(i))+1:(ss*neuron_array(i)))) * diag(w{1,i+1}(a,:)') * kron(ones(neuron_array(i),1),x{1,i}(((ss-1)*neuron_array(i))+1:(ss*neuron_array(i)))'))))];
                end
                jac{i,n+1+i}(((ss-1)*neuron_array(i))+1:ss*neuron_array(i) , : ) = my_arr;
            end
            jac{i,((2*n)+i)} = eye(ns*neuron_array(i,1));
            jac{i,(((2*n)+1)+i)} = -kron(eye(ns),w{1,i+1}') * diag(obj_jac.grad(kron(eye(ns),w{1,i+1})*x{1,i}));
        end
        for s = 1 : ns
            for j = 1 : neuron_array(n)
                jac{n-1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)),((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))) = jac{n-1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)),((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))) - (l{1,n}(((s-1)*neuron_array(n))+j) * obj_jac2.grad2(w{1,n}(j,:)*x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)))) * diag(w{1,n}(j,:)') * kron(ones(neuron_array(n-1),1),w{1,n}(j,:)));
            end
        end
        for ss = 1 : ns
            my_arr = [];
            for a = 1 : neuron_array(n)
                my_arr = [my_arr -(l{1,n}(((ss-1)*neuron_array(n))+a) * ((obj_jac2.grad(w{1,n}(a,:)*x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))) * eye(neuron_array(n-1))) + (obj_jac2.grad2(w{1,n}(a,:)*x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))) * diag(w{1,n}(a,:)') * kron(ones(neuron_array(n-1),1),x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))'))))];
            end
            jac{n-1,2*n}(((ss-1)*neuron_array(n-1))+1:ss*neuron_array(n-1) , : ) = my_arr;
        end
        jac{n-1,((2*n)+n-1)} = eye(ns*neuron_array(n-1,1));
        jac{n-1,(((2*n)+1)+n-1)} = -kron(eye(ns),w{1,n}') * diag(obj_jac2.grad(kron(eye(ns),w{1,n})*x{1,n-1}));

        jac{n,n} = eye(ns*neuron_array(n,1));
        jac{n,(3*n)} = eye(ns*neuron_array(n,1));

        for sp = 1 : ns
            a = zeros(size(jac{(n+1),(n+1)},1),size(jac{(n+1),(n+1)},2));
            for i = 1 : neuron_array(1)
                %jac{(n+1),(n+1)}(i,i) = -l{1,1}(i) * obj_jac.grad2(w{1,1}(i,:)*x0) * diag(x0) * kron(ones(size(x0,1),1),x0');
                a((((i-1)*k0)+1):(i*k0),(((i-1)*k0)+1):(i*k0)) = -l{1,1}(((sp-1)*neuron_array(1))+i) * obj_jac.grad2(w{1,1}(i,:)*x0(((sp-1)*k0)+1:(sp*k0))) * diag(x0(((sp-1)*k0)+1:(sp*k0))) * kron(ones(k0,1),x0(((sp-1)*k0)+1:(sp*k0))');
            end
            jac{n+1,n+1} = jac{n+1,n+1} + a;
        end
    %     for sm = 1 : ns
    %         jac{(n+1),((2*n)+1)}((((sm-1)*k0*neuron_array(1))+1):(sm*k0*neuron_array(1)),(((sm-1)*neuron_array(1))+1):(sm*neuron_array(1))) = -kron(diag(obj_jac.grad(w{1,1}*x0(((sm-1)*k0)+1:(sm*k0)))),x0(((sm-1)*k0)+1:(sm*k0)));
    %     end
        jac{(n+1),((2*n)+1)} = [];
        for sa = 1 : ns
            %a = zeros(neuron_array(1)*k0,neuron_array(1));
            a = -kron(diag(obj_jac.grad(w{1,1}*x0(((sa-1)*k0)+1:sa*k0))),x0(((sa-1)*k0)+1:sa*k0));
            jac{(n+1),((2*n)+1)} = [jac{(n+1),((2*n)+1)} a];
        end
    %     for i = 1 : neuron_array(1)
    %         jac{(n+1),((2*n)+1)}(i,i) = -obj_jac.grad(w{1,1}(i,:)*x0) * x0;
    %     end

        for i = 1 : (n - 2)
            jac{((n+1)+i),i} = jac{i,n+1+i}';

            for sp = 1 : ns
                d = zeros(size(jac{((n+1)+i),((n+1)+i)},1),size(jac{((n+1)+i),((n+1)+i)},2));
                for j = 1 : neuron_array(i+1)
                    d((((j-1)*neuron_array(i))+1):(j*neuron_array(i)),(((j-1)*neuron_array(i))+1):(j*neuron_array(i))) = -l{1,i+1}(((sp-1)*neuron_array(i+1))+j) * obj_jac.grad2(w{1,i+1}(j,:)*x{1,i}(((sp-1)*neuron_array(i))+1:(sp*neuron_array(i)))) * diag(x{1,i}(((sp-1)*neuron_array(i))+1:(sp*neuron_array(i)))) * kron(ones(neuron_array(i),1),x{1,i}(((sp-1)*neuron_array(i))+1:(sp*neuron_array(i)))');
                end
                jac{((n+1)+i),((n+1)+i)} = jac{((n+1)+i),((n+1)+i)} + d;
            end

            jac{((n+1)+i),(((2*n)+1)+i)} = [];
            for sa = 1 : ns
                c = -kron(diag(obj_jac.grad(w{1,i+1}*x{1,i}(((sa-1)*neuron_array(i))+1:sa*neuron_array(i)))),x{1,i}(((sa-1)*neuron_array(i))+1:sa*neuron_array(i)));
                jac{((n+1)+i),(((2*n)+1)+i)} = [jac{((n+1)+i),(((2*n)+1)+i)} c];
            end
        end
        jac{((n+1)+n-1),n-1} = jac{n-1,2*n}';

        for sp = 1 : ns
            d = zeros(size(jac{((n+1)+n-1),((n+1)+n-1)},1),size(jac{((n+1)+n-1),((n+1)+n-1)},2));
            for j = 1 : neuron_array(n)
                d((((j-1)*neuron_array(n-1))+1):(j*neuron_array(n-1)),(((j-1)*neuron_array(n-1))+1):(j*neuron_array(n-1))) = -l{1,n}(((sp-1)*neuron_array(n))+j) * obj_jac2.grad2(w{1,n}(j,:)*x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))) * diag(x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))) * kron(ones(neuron_array(n-1),1),x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))');
            end
            jac{((n+1)+n-1),((n+1)+n-1)} = jac{((n+1)+n-1),((n+1)+n-1)} + d;
        end

        jac{((n+1)+n-1),(((2*n)+1)+n-1)} = [];
        for sa = 1 : ns
            c = -kron(diag(obj_jac2.grad(w{1,n}*x{1,n-1}(((sa-1)*neuron_array(n-1))+1:sa*neuron_array(n-1)))),x{1,n-1}(((sa-1)*neuron_array(n-1))+1:sa*neuron_array(n-1)));
            jac{((n+1)+n-1),(((2*n)+1)+n-1)} = [jac{((n+1)+n-1),(((2*n)+1)+n-1)} c];
        end

        jac{((2*n)+1),1} = eye(ns*neuron_array(1));
        jac{((2*n)+1),n+1} = jac{(n+1),((2*n)+1)}';

        for i = 1 : (n - 2)
            jac{(((2*n)+1)+i),i} = -diag(obj_jac.grad(kron(eye(ns),w{1,i+1})*x{1,i})) * kron(eye(ns),w{1,i+1});
            jac{(((2*n)+1)+i),i+1} = eye(ns*neuron_array(i+1,1));
            jac{(((2*n)+1)+i),n+1+i} = jac{((n+1)+i),(((2*n)+1)+i)}';
        end
        jac{(((2*n)+1)+n-1),n-1} = -diag(obj_jac2.grad(kron(eye(ns),w{1,n})*x{1,n-1})) * kron(eye(ns),w{1,n});
        jac{(((2*n)+1)+n-1),n} = eye(ns*neuron_array(n,1));
        jac{(((2*n)+1)+n-1),n+1+n-1} = jac{((n+1)+n-1),(((2*n)+1)+n-1)}';
    else 
        for s = 1 : ns
            for j = 1 : neuron_array(n)
                jac{n-1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)),((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))) = jac{n-1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)),((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1))) - (l{1,n}(((s-1)*neuron_array(n))+j) * obj_jac2.grad2(w{1,n}(j,:)*x{1,n-1}(((s-1)*neuron_array(n-1))+1:(s*neuron_array(n-1)))) * diag(w{1,n}(j,:)') * kron(ones(neuron_array(n-1),1),w{1,n}(j,:)));
            end
        end
        for ss = 1 : ns
            my_arr = [];
            for a = 1 : neuron_array(n)
                my_arr = [my_arr -(l{1,n}(((ss-1)*neuron_array(n))+a) * ((obj_jac2.grad(w{1,n}(a,:)*x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))) * eye(neuron_array(n-1))) + (obj_jac2.grad2(w{1,n}(a,:)*x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))) * diag(w{1,n}(a,:)') * kron(ones(neuron_array(n-1),1),x{1,n-1}(((ss-1)*neuron_array(n-1))+1:(ss*neuron_array(n-1)))'))))];
            end
            jac{n-1,2*n}(((ss-1)*neuron_array(n-1))+1:ss*neuron_array(n-1) , : ) = my_arr;
        end
        jac{n-1,((2*n)+n-1)} = eye(ns*neuron_array(n-1,1));
        jac{n-1,(((2*n)+1)+n-1)} = -kron(eye(ns),w{1,n}') * diag(obj_jac2.grad(kron(eye(ns),w{1,n})*x{1,n-1}));

        jac{n,n} = eye(ns*neuron_array(n,1));
        jac{n,(3*n)} = eye(ns*neuron_array(n,1));

        for sp = 1 : ns
            a = zeros(size(jac{(n+1),(n+1)},1),size(jac{(n+1),(n+1)},2));
            for i = 1 : neuron_array(1)
                %jac{(n+1),(n+1)}(i,i) = -l{1,1}(i) * obj_jac.grad2(w{1,1}(i,:)*x0) * diag(x0) * kron(ones(size(x0,1),1),x0');
                a((((i-1)*k0)+1):(i*k0),(((i-1)*k0)+1):(i*k0)) = -l{1,1}(((sp-1)*neuron_array(1))+i) * obj_jac.grad2(w{1,1}(i,:)*x0(((sp-1)*k0)+1:(sp*k0))) * diag(x0(((sp-1)*k0)+1:(sp*k0))) * kron(ones(k0,1),x0(((sp-1)*k0)+1:(sp*k0))');
            end
            jac{n+1,n+1} = jac{n+1,n+1} + a;
        end
        
        jac{(n+1),((2*n)+1)} = [];
        for sa = 1 : ns
            %a = zeros(neuron_array(1)*k0,neuron_array(1));
            a = -kron(diag(obj_jac.grad(w{1,1}*x0(((sa-1)*k0)+1:sa*k0))),x0(((sa-1)*k0)+1:sa*k0));
            jac{(n+1),((2*n)+1)} = [jac{(n+1),((2*n)+1)} a];
        end 
        jac{((n+1)+n-1),n-1} = jac{n-1,2*n}';

        for sp = 1 : ns
            d = zeros(size(jac{((n+1)+n-1),((n+1)+n-1)},1),size(jac{((n+1)+n-1),((n+1)+n-1)},2));
            for j = 1 : neuron_array(n)
                d((((j-1)*neuron_array(n-1))+1):(j*neuron_array(n-1)),(((j-1)*neuron_array(n-1))+1):(j*neuron_array(n-1))) = -l{1,n}(((sp-1)*neuron_array(n))+j) * obj_jac2.grad2(w{1,n}(j,:)*x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))) * diag(x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))) * kron(ones(neuron_array(n-1),1),x{1,n-1}(((sp-1)*neuron_array(n-1))+1:(sp*neuron_array(n-1)))');
            end
            jac{((n+1)+n-1),((n+1)+n-1)} = jac{((n+1)+n-1),((n+1)+n-1)} + d;
        end

        jac{((n+1)+n-1),(((2*n)+1)+n-1)} = [];
        for sa = 1 : ns
            c = -kron(diag(obj_jac2.grad(w{1,n}*x{1,n-1}(((sa-1)*neuron_array(n-1))+1:sa*neuron_array(n-1)))),x{1,n-1}(((sa-1)*neuron_array(n-1))+1:sa*neuron_array(n-1)));
            jac{((n+1)+n-1),(((2*n)+1)+n-1)} = [jac{((n+1)+n-1),(((2*n)+1)+n-1)} c];
        end

        jac{((2*n)+1),1} = eye(ns*neuron_array(1));
        jac{((2*n)+1),n+1} = jac{(n+1),((2*n)+1)}';
        
        jac{(((2*n)+1)+n-1),n-1} = -diag(obj_jac2.grad(kron(eye(ns),w{1,n})*x{1,n-1})) * kron(eye(ns),w{1,n});
        jac{(((2*n)+1)+n-1),n} = eye(ns*neuron_array(n,1));
        jac{(((2*n)+1)+n-1),n+1+n-1} = jac{((n+1)+n-1),(((2*n)+1)+n-1)}';
    end
end

