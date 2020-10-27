function vec = make_vector(input_cell)    
    vec = [];
    for i = 1 : size(input_cell,1)
        vec = [vec;input_cell{i,1}];
    end