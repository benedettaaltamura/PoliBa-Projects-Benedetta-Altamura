function [fn_array] = compute_fn_pop(pop, I_min, I_max,x_train,x_test,t_train,t_test)
rng('default')
fn_array = [];
ind = pop{1};
numero_geni = numel(ind);
numero_ind = numel(pop);
for j = 1:numero_ind  %per ogni individuo viene calcolata la funzione ob
    ind = pop{j};

    %fn = compute_fn_ANN(ind, I_min, I_max,x_train,x_test,t_train,t_test);
    fun = ind2fun(ind, I_min, I_max); %hidden layer
    hiddenLayer = round(fun);
    
    k=10;
    C_ANN= zeros(2,2);
    
    trainFcn = 'traingdx';
    performFcn = 'crossentropy';
    net = patternnet(hiddenLayer,trainFcn,performFcn);
    net.layers{end}.transferFcn = 'logsig';
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.75;
    net.divideParam.valRatio = 0.25;
    net.divideParam.testRatio = 0.00;
    net = configure(net,x_train{1},t_train{1});
    net.trainParam.epochs = 500;
    net.trainParam.lr = 1e-3;
    net.trainParam.max_fail = 100;
    
    % Training della rete e valutazione delle performance
    for i = 1:k
        
        %RETE NEURALE ARTIFICIALE
        net=init(net);
    
        [trained_net,tr] = train(net,x_train{i},t_train{i});
        y_pred_ANN{i} = trained_net(x_test{i});
        y_pred_bin_ANN{i} = double(y_pred_ANN{i} > 0.5);
        cm_ANN=confusionmat(t_test{i},y_pred_bin_ANN{i});
      
     
        accurracy_ANN(i) = (cm_ANN(1,1)+cm_ANN(2,2))/sum(cm_ANN,'all');
        
        C_ANN(1,1) = C_ANN(1,1) + cm_ANN(1,1);
        C_ANN(2,2) = C_ANN(2,2) + cm_ANN(2,2);
        C_ANN(2,1) = C_ANN(2,1) + cm_ANN(2,1);
        C_ANN(1,2) = C_ANN(1,2) + cm_ANN(1,2);
    end
    ACC_test = mean(accurracy_ANN);
    fitness = ACC_test; %valore funzione di fitness per ciascun individuo
    
    fn_array(1, end+1) = fitness; %array in cui ogni valore è associato ad un individuo della pop
end

end





function [fun] = ind2fun(ind, I_min, I_max)
fun = [];
num_geni = numel(ind);

for i = 1:num_geni
    fun(i) = bit2double2(ind{i}, I_min, I_max);
end
%fprintf("fun = ");
%disp(fun)
end


function [out]= bit2double2(bit_array, I_min,I_max)

num_bit= numel(bit_array);
decimale= 0;
for i =1:num_bit
    decimale= decimale+ bit_array(i) * 2^ (num_bit-i);

end

step= (I_max - I_min) / (2^num_bit-1); %passettino a ogni bit
out= I_min + step* decimale;
end
