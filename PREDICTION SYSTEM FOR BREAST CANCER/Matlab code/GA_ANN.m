% GA ANN
function [hl12]= GA_ANN(x_train,x_test,t_train,t_test)
numero_geni = 2;
numero_bit = 7;
intervallo_min = 2;
intervallo_max = 129;
numero_individui = 10;
prob_mut = 0.2; %se l'aumento l'individuo massimo diventa migliore perche creo diversità . l'indviduo medio sarà peggiore
num_epochs = 40;
elitism = 1;  %individui che devono essere preservati della generazione precedente


%fprintf("numero_geni      = %d\n", numero_geni);
%fprintf("numero_bit       = %d\n", numero_bit);
%fprintf("intervallo_min   = %d\n", intervallo_min);
%fprintf("intervallo_max   = %d\n", intervallo_max);
%fprintf("numero_individui = %d\n", numero_individui);
%fprintf("prob_mut         = %d\n", prob_mut);
%fprintf("num_epochs       = %d\n", num_epochs);
%fprintf("elitism          = %d\n", elitism);



%% Creazione di una popolazione (M individui)
%fprintf("\nGenerazione Popolazione\n");
pop = {};
for i = 1:numero_individui
    ind={}; %inizializzo cell arrey
    for j=1:numero_geni
        ind{j}= randi([0,1],1,numero_bit); %a ogni ciclo aggiungo un elemento al cell arrey      
    end
    pop{i} = ind;    
end
%30 elementi


%% Ottimizzazione
best_epoch = []; %salvo migliore medio e peggiore per capire come sta andando l'ottimizzazione
avg_epoch = [];
worst_epoch = [];
for epoch_idx = 1:num_epochs
    epoch_idx
    %epoch_idx
    new_pop = {}; 
    
    fn_array = compute_fn_pop(pop, intervallo_min, intervallo_max,x_train,x_test,t_train,t_test);
    
    [~, idx] = sort(fn_array, 'descend');
    for ind_idx = 1:elitism %uso gli indici ordinati per prendere gli elemeti che devo ricopiare della vecchia pop
        new_pop{ind_idx} = pop{idx(ind_idx)}; %avrà individui della vecchia pop
    end
    
    %il resto degli individui sono selezionati partendo dai due genitori 
    for ind_idx = elitism+1 :numero_individui
        idx1 = choose_ind(fn_array); 
        idx2 = choose_ind(fn_array);
        while idx1 == idx2  %per evitare di scegliere due individui uguali 
           idx2 = choose_ind(fn_array);
        end

        genitore1 = pop{idx1}; %sui due individui genitire è fatto il cross
        genitore2 = pop{idx2};

        figlio = reproduce(genitore1, genitore2);
        
        %mutazione figlio
        if rand() < prob_mut %rand() restituisce un numero a caso tra 0 e 1. se è < di prob_min il figlio è mutato       
            figlio_m = {};
            num_geni = numel(figlio);
            for i = 1:num_geni
                gene_figlio = figlio{i};
                len_gene = numel(gene_figlio);          %trovo punto dove mutare
                mut_point = randi([1,len_gene]);
                gene_figlio(mut_point) = double(not(logical(gene_figlio(mut_point))));
                gene_m = gene_figlio;
                figlio_m{end+1} = gene_m;
            end
            figlio=figlio_m;
        end

        new_pop{ind_idx} = figlio;
    end

    % visualizzo individui dell'epoca i-esima 
%     fprintf("individui all'epoca: %d \n",epoch_idx);
%     for c= 1: numel(pop)
%         individuo= ind2fun(pop{c}, intervallo_min, intervallo_max);
%         disp(individuo)
%     end


    pop = new_pop;
    %fprintf("Iter %d / %d\n", epoch_idx, num_epochs);
    %fprintf("Fitness array = ");
    %disp(fn_array);

    best_fn = max(fn_array);
    worst_fn = min(fn_array);
    avg_fn = mean(fn_array);
    best_epoch(end+1) = best_fn;
    avg_epoch(end+1) = avg_fn;
    worst_epoch(end+1) = worst_fn;
    
  
    
end

[best_fn, best_idx] = max(best_epoch);
best_ind = pop{best_idx};
fun = ind2fun(best_ind, intervallo_min, intervallo_max);
fprintf("best fitness = %.3f \n miglior individuo trovato all'epoca %.0f \n hiddenlayer=", best_fn, best_idx);
disp(fun)
fprintf("\n");

% Plot Trends
X = linspace(1, num_epochs, num_epochs);
figure
hold on

title("Trends during the optimization process")
plot(X, best_epoch, '-o', ...
     X, avg_epoch, ':*', ...
     X, worst_epoch, ':d')
legend("Best", "Average", "Worst")
hold off

hl12=fun;

end


function [chosen_idx] = choose_ind(fn_array)
p = fn_array / sum(fn_array); %chi ha la fitness più alta ha piu possibilita di essere scelto 
cs = cumsum(p);
rand_v = rand();
csp = cs(cs>rand_v);
chosen = csp(1);
chosen_idx = find(cs==chosen);
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


%CROSS-OVER SUL CROMOSOMA

function [figlio] = reproduce(genitore1,genitore2)

figlio={};
numero_geni= numel(genitore1);
gene1= genitore1{1};
numero_bit= numel(gene1);

%converto i geni in cromosomi dei 2 genitori
cromosoma1=[];
numero_geni=numel(genitore1);
for i=1: numero_geni
    cromosoma1= [cromosoma1, genitore1{i}]; %concatena il primo gene e alla seconda iterazione il secondo
end

cromosoma2=[];
numero_geni=numel(genitore2);
for i=1: numero_geni
    cromosoma2= [cromosoma2, genitore2{i}]; %concatena il primo gene e alla seconda iterazione il secondo
end


%cross over a singolo punto
numero_bit = numel(cromosoma1);
idx_sep = randi([2,numero_bit-1]);
cx = [];
for i = 1:idx_sep
    cx(i) = cromosoma1(i);
end
for i = idx_sep+1 :numero_bit
    cx(i) = cromosoma2(i);
end
cromosomax=cx;

% converto il cromosoma in geni
for i = 1:numero_geni
    j = i + (numero_bit-1) * (i-1); 
    %fprintf("Interval = [%d %d]\n", j, j+numero_bit-1)
    figlio{i} = [cromosomax(j:j+numero_bit-1)];
end

end



