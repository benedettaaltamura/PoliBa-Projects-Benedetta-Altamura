%% CARICO I FILE INIZIALI

clear all 
close all
clc
rng('default')

% caricamento dei counts
data_table = readtable('data_table.csv', ...
                                 'ReadRowNames', true, ...
                                 'VariableNamingRule','preserve');

% caricamento features cliniche
clinical_data= readtable('data_table_clinical2.csv','ReadRowNames', true, ...
                                 'VariableNamingRule','preserve');


% elimino righe con valori NaN
missing_values = ismissing(clinical_data);
sum_missing_values = sum(missing_values, 2);
missing_rows = find(sum_missing_values > 0);
clinical_data(missing_rows, :) = [];
data_table(:, missing_rows) = [];


% target --> PRIMARY DIAG
target= clinical_data(:,8);
% eta=clinical_data(:,1);
% race=clinical_data(:,7);
% stadio=clinical_data(:,3);


%indici aggiornati 
labels_2=find(target.primary_diag =="Infiltrating duct carcinoma, NOS"); %indice dei pazienti con stadio 1
labels_1=find(target.primary_diag =="Lobular carcinoma, NOS"); %indice dei pazienti con stadio 2


% 1=infiltrating ; 0=lobular
target.primary_diag = target.primary_diag=="Infiltrating duct carcinoma, NOS";
clinical_data.primary_diag= clinical_data.primary_diag=="Infiltrating duct carcinoma, NOS";


%trasformo tutto in array
data_table_array=table2array(data_table);
clinical_data= clinical_data(:,[2,7]);
clinical_data= table2array(clinical_data);
target=table2array(target);


%% CARATTERISTICHE DEI COUNTS

% histogram of the counts for a single sample
figure
histogram(data_table_array(:,10));
title('Histogram of the counts for a single sample');
xlabel('Row Expression Counts');
ylabel('Number of genes')


% histogram of the counts for all samples
figure
histogram(data_table_array);
title('Data Distribution');
xlabel('Counts data');
ylabel('Absolute Frequency')
xlim([-10,100000])

%% NORMALIZZAZIONE
% NORMALIZZAZIONE CON SIZE FACTOR

counts = data_table_array ;
% estimate pseudo-reference with geometric mean row by row
pseudoRefSample = geomean(counts,2);
nz = pseudoRefSample > 0;
ratios = bsxfun(@rdivide,counts(nz,:),pseudoRefSample(nz));
sizeFactors = median(ratios,1);
% transorm to common scale
normCounts = bsxfun(@rdivide,counts,sizeFactors);

% Boxplots
figure
    subplot(2,1,1)
        maboxplot(log2(counts(:,1:5)+1),'title','Raw Read Count','orientation','horizontal')
        ylabel('sample');
        xlabel('log2(counts)');
    subplot(2,1,2)
        maboxplot(log2(normCounts(:,1:5)+1),'title','Normalized Read Count','orientation','horizontal')
        ylabel('sample');
        xlabel('log2(counts)');


 % NORMALIZZAZIONE LOG2
normCounts= log2(normCounts +1);
figure
histogram(normCounts);
title( 'Data Normalized with log2')
xlabel('log2(counts+1)');
ylabel('Absolute Frequency');
%%  riduzione dimensionalità con PCA 

var_p=80;

[coeff,score,latent,tsquared, ...
    explained,mu] = pca(normCounts', 'Centered', true);
sum_exp= cumsum(explained);
n= find( sum_exp>var_p , 1);


x_pca = normCounts' *coeff;
    n_comp = numel(explained);
    for ii = 1:n_comp
        sum_ = sum(explained(1:ii));
        if sum_ > var_p,
            break; 
        end
    end
    num_pc = ii;

    fprintf("For preserving %.2f%s of variance, you have to use %d PC\n", var_p, "%", num_pc);
    x_pca__reduced= x_pca(:, 1:n);

    figure
        bar(explained/100)
        xlim([0 num_pc])
        title(sprintf("Variance to explain = %.2f",var_p/100))
        xlabel(sprintf("First %d Principal Component",num_pc))
        ylabel("Variance explained")


figure
scatter(score(labels_1,1), ...
        score(labels_1,2),5, 'black','filled');
hold on
scatter(score(labels_2,1), ...
       score(labels_2,2),5, 'red','filled');


title ("PCA")
xlabel("PCA-1")
ylabel("PCA-2")
legend('Lobular carcinoma','Infiltrating duct carcinoma','Location','southwest');
hold off


figure
scatter3(score(labels_1,1), ...
        score(labels_1,2),...
        score(labels_1,3),5, 'black','filled');
hold on
scatter3(score(labels_2,1), ...
       score(labels_2,2),...
       score(labels_2,3),5, 'red','filled');

title ("PCA")
xlabel("PCA-1")
ylabel("PCA-2")
zlabel("PCA-3")
legend('Lobular carcinoma','Infiltrating duct carcinoma','Location','northeast');
hold off


%% tSNE
% riduzione con t sne
genDataEmbedding = tsne(normCounts', ...
                        'Algorithm', 'exact', ...
                        'NumDimensions', 2);
figure
scatter(genDataEmbedding(labels_1,1), ...
        genDataEmbedding(labels_1,2), ...
        5, 'black');
hold on
scatter(genDataEmbedding(labels_2,1), ...
       genDataEmbedding(labels_2,2), ...
       5, 'red');


title ("tSNE")
xlabel("tSNE-1")
ylabel("tSNE-2")
%saveas(gcf, "TSNE.png")

hold off


%% bilanciamento 
doBilanciamento= true;

if doBilanciamento

    counts_clinic= [x_pca__reduced';clinical_data'];
    % Oversampling labels 1
    AAA= counts_clinic(:,labels_1)';
    AX=resample(AAA,239,161);
    [row col]=size(AX);
    AX=[counts_clinic(:,labels_1), AX'];
    AX(end-1:end,:)=round(AX(end-1:end,:));

    % Check race and age at diagnosis dummy values
    for i=1:size(AX,2)
        temp=AX(end,i);
        if temp==0
            AX(end,i)=1;
        elseif temp==5
            AX(end,i)=4;
        end
    end

    % Undersampling labels_2
    [labels_3]=datasample(labels_2,400,'Replace',false);
    AX2= counts_clinic(:,labels_3);

    %new data_table_arrey and clinical data with balancing
    counts_clinic_finale= [AX,AX2];
    x_pca__reduced = counts_clinic_finale(1:size(x_pca__reduced,2),:);
    clinical_data= round(counts_clinic_finale((size(x_pca__reduced,1)+1):end,:));
    clinical_data= clinical_data';

    %new target--> from 1 to 400  labels_1 --> 0
    %              from 401 to 800  labels_2 --> 1
    target= [zeros(400,1); ones(400,1)];

    %new labels
    labels_1_finale= find(target==0);
    labels_2_finale= find(target==1);

    x_pca__reduced= x_pca__reduced';

    % PLOT NUMBER OF PATIENTS PER CLASS AFTER BALANCING
    figure
    y=length(labels_2);
    y2 = categorical({'Before'});
    bar(y2,y);
    hold on
    y3=length(labels_2_finale);
    y4 = categorical({'After'});
    bar(y4,y3);
    legend('Number of patients before balancing','Number of patients before and after balancing');
    ylim([0 900])
    title("Number of patients in class 1 before and after balancing")
    hold off


    figure
    y=length(labels_1);
    y2 = categorical({'Before'});
    bar(y2,y);
    hold on
    y3=length(labels_1_finale);
    y4 = categorical({'After'});
    bar(y4,y3);
    legend('Number of patients before balancing','Number of patients before and after balancing');
    ylim([0 900])
    title("Number of patients in class 2 before and after balancing")
    hold off
   
    figure
    y=length(labels_2_finale);
    y2 = categorical({'Classe 1'});
    bar(y2,y);
    hold on
    y3=length(labels_1_finale);
    y4 = categorical({'Classe 2'});
    bar(y4,y3);
    legend('Patients with Infiltrating duct carcinoma','Patients with Lobular carcinoma');
    ylim([0 900])
    title("Number of patients in each class after balancing")
    hold off
end
%% classificatore con CROSS VALIDAZIONE
rng('default')
Xpval = x_pca__reduced';
clinical_feat = clinical_data';
target_feat= target;
Xs = [Xpval; clinical_feat];
tp = double(target_feat'); %target

k = 10;
cv = cvpartition(tp,'Kfold',k,'stratify',true);
C_ANN = zeros(2,2);
C_LR = zeros(2,2);
C_DT=zeros(2,2);
for i = 1:k
    idx = training(cv,i);
    idx_train = find(idx);
    idx_test= find(~idx);
    x_train{i} = Xs(:,idx_train);
    x_test{i} = Xs(:,idx_test);
    t_train{i} = tp(:,idx_train);
    t_test{i} = tp(:,idx_test);
end

%hiddenLayer= GA_ANN(x_train,x_test,t_train,t_test);
hiddenLayer=[55 98];
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
    net=init(net);
    [trained_net,tr] = train(net,x_train{i},t_train{i});
    y_pred_ANN{i} = trained_net(x_test{i});
    y_pred_bin_ANN{i} = double(y_pred_ANN{i} > 0.5);
    cm_ANN=confusionmat(t_test{i},y_pred_bin_ANN{i});

    accurracy_ANN(i) = (cm_ANN(1,1)+cm_ANN(2,2))/sum(cm_ANN,'all');
    precision_ANN(i)= (cm_ANN(2,2)/(cm_ANN(2,2)+cm_ANN(1,2)));
    recall_ANN(i)= (cm_ANN(2,2)/(cm_ANN(2,2)+cm_ANN(2,1)));
    miss_rate_ANN(i)= (cm_ANN(2,1)/(cm_ANN(2,2)+cm_ANN(2,1)));
    % cm11 = true negative
    % cm22 = true positive
    % cm12 = false positive
    % cm21 = false negative
    C_ANN(1,1) = C_ANN(1,1) + cm_ANN(1,1);
    C_ANN(2,2) = C_ANN(2,2) + cm_ANN(2,2);
    C_ANN(2,1) = C_ANN(2,1) + cm_ANN(2,1);
    C_ANN(1,2) = C_ANN(1,2) + cm_ANN(1,2);

    [fpr_ANN{i}, tpr_ANN{i}, th_ANN{i} ,auc_ANN(i)] = perfcurve (t_test{i},y_pred_ANN{i},1);

     %REGRESSIONE LINEARE Logistica 
     t_train{i}= logical(t_train{i});
     t_test{i}= logical(t_test{i});
     LR= fitclinear(x_train{i}',t_train{i}','Learner','logistic');
     [y_pred_LR{i}, score_LR{i}] = LR.predict(x_test{i}');
     cm_LR=confusionmat(t_test{i},y_pred_LR{i});

     accurracy_LR(i) = (cm_LR(1,1)+cm_LR(2,2))/sum(cm_LR,'all');
     precision_LR(i)= (cm_LR(2,2)/(cm_LR(2,2)+cm_LR(1,2)));
     recall_LR(i)= (cm_LR(2,2)/(cm_LR(2,2)+cm_LR(2,1)));
     miss_rate_LR(i)= (cm_LR(2,1)/(cm_LR(2,2)+cm_LR(2,1)));
    % cm11 = true negative
    % cm22 = true positive
    % cm12 = false positive
    % cm21 = false negative
     C_LR(1,1) = C_LR(1,1) + cm_LR(1,1);
     C_LR(2,2) = C_LR(2,2) + cm_LR(2,2);
     C_LR(2,1) = C_LR(2,1) + cm_LR(2,1);
     C_LR(1,2) = C_LR(1,2) + cm_LR(1,2);
     y_pred_bin_LR= score_LR{i};
     [fpr_LR{i}, tpr_LR{i}, th_LR{i} ,auc_LR(i)] = perfcurve (t_test{i},y_pred_bin_LR(:,2)',1);

     % DECISION TREE
     DT = fitctree(x_train{i}',t_train{i}');
     [y_pred_DT{i}, score_DT{i}] = predict(DT,x_test{i}');
     cm_DT=confusionmat(t_test{i},y_pred_DT{i});
     accurracy_DT(i) = (cm_DT(1,1)+cm_DT(2,2))/sum(cm_DT,'all');
     precision_DT(i)= (cm_DT(2,2)/(cm_DT(2,2)+cm_DT(1,2)));
     recall_DT(i)= (cm_DT(2,2)/(cm_DT(2,2)+cm_DT(2,1)));
     miss_rate_DT(i)= (cm_DT(2,1)/(cm_DT(2,2)+cm_DT(2,1)));
     C_DT(1,1) = C_DT(1,1) + cm_DT(1,1);
     C_DT(2,2) = C_DT(2,2) + cm_DT(2,2);
     C_DT(2,1) = C_DT(2,1) + cm_DT(2,1);
     C_DT(1,2) = C_DT(1,2) + cm_DT(1,2);
     y_pred_bin_DT= score_DT{i};
     [fpr_DT{i}, tpr_DT{i}, th_DT{i} ,auc_DT(i)] = perfcurve (t_test{i},y_pred_bin_DT(:,2)',1);
%      if i==3
%          view(DT,"Mode","graph")
%      end

end 


% RETE NEURALE ARTIFICIALE 
ACC_ANN = mean(accurracy_ANN);
PRE_ANN= mean(precision_ANN);
MISS_ANN= mean(miss_rate_ANN);
REC_ANN= mean(recall_ANN);
AUC_ANN = mean(auc_ANN);

figure
confusionchart(C_ANN);
title("matrice di confusione cross-validazione 10-folds rete ANN");
fprintf("Risultati rete neurale artificiale con cross-validazione 10-folds: \n")
fprintf("Accuratezza media: %.2f \n",ACC_ANN*100)
fprintf("Precisione media: %.2f \n",PRE_ANN*100)
fprintf("Miss Rate media: %.2f \n",MISS_ANN*100)
fprintf("Recall media: %.2f \n",REC_ANN*100)
fprintf("Area media sottesa alla curva roc: %.2f \n",AUC_ANN*100)
fprintf("\n");

% REGRESSIONE LOGISTICA 
ACC_LR = mean(accurracy_LR);
PRE_LR= mean(precision_LR);
MISS_LR= mean(miss_rate_LR);
REC_LR= mean(recall_LR);
AUC_LR = mean(auc_LR);
figure
confusionchart(C_LR);
title("Matrice di confusione cross-validazione 10-folds LR");
fprintf("Risultati regressione logistica con cross-validazione 10-folds: \n")
fprintf("Accuratezza media: %.2f \n",ACC_LR*100)
fprintf("Precisione media: %.2f \n",PRE_LR*100)
fprintf("Miss Rate media: %.2f \n",MISS_LR*100)
fprintf("Recall media: %.2f \n",REC_LR*100)
fprintf("Area media sottesa alla curva roc: %.2f \n",AUC_LR*100)
fprintf("\n");

% DECISION TREE
ACC_DT = mean(accurracy_DT);
PRE_DT= mean(precision_DT);
MISS_DT= mean(miss_rate_DT);
REC_DT= mean(recall_DT);
AUC_DT = mean(auc_DT);

figure
confusionchart(C_DT);
title("Matrice di confusione Albero Decisionale");
fprintf("Risultati Albero Decisionale con cross-validazione 10-folds: \n")
fprintf("Accuratezza media Albero Decisionale: %.2f \n",ACC_DT*100)
fprintf("Precisione media Albero Decisionale: %.2f \n",PRE_DT*100)
fprintf("Miss Rate media Albero Decisionale: %.2f \n",MISS_DT*100)
fprintf("Recall media Albero Decisionale: %.2f \n",REC_DT*100)
fprintf("Area media sottesa alla curva roc Albero Decisionale: %.2f \n",AUC_DT*100)
fprintf("\n");


%% LA CURVA ROC PER ANN
FPR_ANN=[];
for i=1:length(fpr_ANN)
    FPR_ANN=[FPR_ANN fpr_ANN{i}];
end

TPR_ANN=[];
for i=1:length(tpr_ANN)
    TPR_ANN=[TPR_ANN tpr_ANN{i}];
end

TH_ANN=[];
for i=1:length(th_ANN)
    TH_ANN=[TH_ANN th_ANN{i}];
end


FPR_ANN_mean=mean(FPR_ANN,2);
TPR_ANN_mean=mean(TPR_ANN,2);
TH_ANN_mean=mean(TH_ANN,2);

dists_now = [];
num_th = numel(th_ANN{i});
for j = 1:num_th
    dists_now(end+1) = FPR_ANN_mean(j) ^ 2 + (TPR_ANN_mean(j)-1) ^ 2;
end
[vmin, imin] = min(dists_now);

soglia=TH_ANN_mean(imin);
cord_x=FPR_ANN_mean(imin);
cord_y=TPR_ANN_mean(imin);
fprintf("Il best cut-off per l'ANN è: %.2f \n",soglia)
fprintf("Si ottiene in corrispondenza della cordinata x: %.2f e della coordinata y:%.2f \n",cord_x,cord_y)
fprintf("\n");
 
figure
plot([0,cord_x],[1 cord_y],'-o','LineWidth',1)
hold on
plot(FPR_ANN,TPR_ANN)
hold on
plot(FPR_ANN_mean,TPR_ANN_mean,'r','LineWidth',2)
hold on
plot([0 1],[0 1],'r','LineStyle','--')
title("ROC Curve for ANN, Mean AUC = ", AUC_ANN)
legend("Min Distance","ROC fold 1","ROC fold 2","ROC fold 3","ROC fold 4","ROC fold 5","ROC fold 6","ROC fold 7","ROC fold 8","ROC fold 9","ROC fold 10","Mean ROC","Chance",'Location','southeast');
hold off


%% LA CURVA ROC PER LA LR 
FPR_LR=[];
for i=1:length(fpr_LR)
    FPR_LR=[FPR_LR fpr_LR{i}];
end

TPR_LR=[];
for i=1:length(tpr_LR)
    TPR_LR=[TPR_LR tpr_LR{i}];
end

TH_LR=[];
for i=1:length(th_LR)
    TH_LR=[TH_LR th_LR{i}];
end

FPR_LR_mean=mean(FPR_LR,2);
TPR_LR_mean=mean(TPR_LR,2);
TH_LR_mean=mean(TH_LR,2);

dists_now = [];
num_th = numel(th_LR{i});
for j = 1:num_th
    dists_now(end+1) = FPR_LR_mean(j) ^ 2 + (TPR_LR_mean(j)-1) ^ 2;
end
[vmin, imin] = min(dists_now);

soglia=TH_LR_mean(imin);
cord_x=FPR_LR_mean(imin);
cord_y=TPR_LR_mean(imin);
fprintf("Il best cut-off per la LR è: %.2f \n",soglia)
fprintf("Si ottiene in corrispondenza della cordinata x: %.2f e della coordinata y:%.2f \n",cord_x,cord_y)
fprintf("\n");

figure
plot([0,cord_x],[1 cord_y],'-o','LineWidth',1)
hold on
plot(FPR_LR,TPR_LR)
hold on
plot(FPR_LR_mean,TPR_LR_mean,'r','LineWidth',2)
hold on
plot([0 1],[0 1],'r','LineStyle','--')
title("ROC Curve for LR, Mean AUC = ", AUC_LR)
legend("Min Distance","ROC fold 1","ROC fold 2","ROC fold 3","ROC fold 4","ROC fold 5","ROC fold 6","ROC fold 7","ROC fold 8","ROC fold 9","ROC fold 10","Mean ROC","Chance",'Location','southeast');
hold off

%% BINARY DECISION TREE
figure
for i=1:k
    plot(fpr_DT{i}, tpr_DT{i});
    hold on
    
end
title("Curve ROC Albero Decisionale");
hold on
plot([0 1],[0 1],'r','LineStyle','--')
legend("ROC fold 1","ROC fold 2","ROC fold 3","ROC fold 4","ROC fold 5","ROC fold 6","ROC fold 7","ROC fold 8","ROC fold 9","ROC fold 10","Chance",'Location','southeast');
hold off
