%%data retrieval 
clc

import matlab.net.*  
import matlab.net.http.*

base_url= "https://api.gdc.cancer.gov/";

%% files 
files_url= strcat(base_url, "files"); 
method= RequestMethod.POST;
filter= fileread('filter.json'); 
filter= jsondecode(filter); 
body= MessageBody(filter);
uri= URI(files_url);
request= RequestMessage(method, [], body);
[response, ~, ~]= send(request, uri); 

%% data
data= response.Body.Data.data.hits; 

%codice paziente
caso= data.cases;
caso= caso.case_id;
 
if isstruct(data)
    data2= cell(length(data),1);
    for i= 1 : length(data)
        data2(i)= {data(i)};  
    end
else 
    data2= data; 
end

%% creo cartella con i file 
mkdir("./files")
if isfolder("./files") 
    delete("./files/*.tsv");
else
    mkdir("./files");
end


%% download files
options= weboptions('Timeout', 60);

data_url= strcat(base_url, "data"); 
for i= 1: length(data2)
    temp= data2{i};
    fid= temp.file_id;
    name= temp.file_name;
    file_url= strcat(data_url, "/", fid);
    file_path= fullfile("./files",name); 
    websave(file_path, file_url, options);
end


%% files con geni di interesse 

files_url= strcat(base_url, "genes"); 
method= RequestMethod.POST;
filter= fileread('genes.2023-01-23.json'); 
filter= jsondecode(filter);
body_gene= MessageBody(filter);
uri= URI(files_url);
request_gene= RequestMessage(method, [], body_gene);
[response_gene, altro, altro1]= send(request_gene, uri);

%% data geni
data_gene= response_gene.Body.Data.data.hits; 

if isstruct(data_gene)
    data2_gene= cell(length(data_gene),1);
    for i= 1 : length(data_gene)
        data2_gene(i)= {data_gene(i)};
    end
else 
    data2_gene= data_gene; 
end

gene_name = struct2table(filter);
gene_name=gene_name.symbol;
gene_name=cell2table(gene_name);


%% processo i file 
base_file_path= "./files";
data_table= table();        %inserisco i conteggi cioe i counts
info_table= table();        %per ogni trascritto inserisco : nome  e funzione 
target_table = table();     %diagnosi per ogni soggetto
codici_pazienti= table();   %codici pazienti 
first= true; 
data_table_clinical= table();

for i =  1: length(data2)
    temp= data2{i}; 

    name= temp.file_name;  
    temp_file_name= fullfile(base_file_path, name);

    if isfield(temp, 'cases')

     if isfield(temp.cases , 'diagnoses')
         if isfield(temp.cases.diagnoses, 'ajcc_pathologic_stage')

        fid= temp.cases.case_id ;   
        counts_data= readtable(temp_file_name, 'FileType', 'text');

        %features cliniche
        paziente= i
        diag= temp.cases.diagnoses.morphology;
        ajcc_pathologic_t= temp.cases.diagnoses.ajcc_pathologic_t;
        ajcc_pathologic_m= temp.cases.diagnoses.ajcc_pathologic_m;
        ajcc_pathologic_n= temp.cases.diagnoses.ajcc_pathologic_n;
        ajcc_pathologic_stage= temp.cases.diagnoses.ajcc_pathologic_stage;
        primary_diag= temp.cases.diagnoses.primary_diagnosis;
        age_at_diag= temp.cases.diagnoses.age_at_diagnosis;
        race= temp.cases.demographic.race;
        age= temp.cases.demographic.age_at_index;
        treatments=temp.cases.diagnoses.treatments.treatment_or_therapy;
        
        %elimino le parti inutili della tabella  
        counts_data= counts_data(5:end, 1:4); 
        %seleziono i geni codificanti proteine 
        idx = find( counts_data.gene_type=="protein_coding");
        counts_data= counts_data(idx,:);

        %elimino i duplicati 
        [C, ia, ic]= unique(counts_data.gene_name, "stable");
        counts_data2= counts_data(ia,:);

        %seleziono solo i count dei geni di interesse del singolo paziente
        counts_data3=cell2table(counts_data2.gene_name);
        counts_data3_a=table2array(counts_data3);
        counts_data3_c=categorical(counts_data3_a);
        gene_name_a=table2array(gene_name);
        gene_name_c=categorical(gene_name_a);
        indici=ismember(counts_data3_c,gene_name_c);
        idx_geni_di_interesse=find(indici==1); 
        counts_data_finale= counts_data2(idx_geni_di_interesse,:);
       
        
        %creo tabella finale con tutti i counts_data_finale di tutti i pazienti
        data_table.(fid)= counts_data_finale.unstranded;  %ogni colonna di data table è identificabile dal numero e dal file id del soggetto.    
      
        data_table_clinical.(fid)= {age;age_at_diag; ajcc_pathologic_stage; ajcc_pathologic_n; ajcc_pathologic_t; ajcc_pathologic_m; race; primary_diag; diag; treatments };
        nome_righe_data_clinici= {'age','age_at_diag', 'ajcc_pathologic_stage', 'ajcc_pat_n', 'ajcc_pathologic_t', 'ajcc_pat_m', 'race', 'primary_diag', 'diag','treatments'};


        %nome righe e colonna tabella
        if first
            %strutto i valori per assegnare nomi alle righe della tabella
            data_table.Properties.RowNames= counts_data_finale.gene_name; 
            data_table_clinical.Properties.RowNames= nome_righe_data_clinici;
            info_table.('name')= counts_data_finale.gene_name;
            info_table.('function')= counts_data_finale.gene_type;
            info_table.Properties.RowNames= counts_data_finale.gene_id;
            first= false;  %questo mi permette di fare questo if solo una volta 
        end

        else 
             
            delete(temp_file_name);
        end
        
     else
         delete(temp_file_name);
     
     end  
    
    else 
        delete(temp_file_name);
    end
         
end

%% creo la tabella finale che contiene sulle righe i pazienti e sulle colonne features+geni

clinical_data=rows2vars(data_table_clinical);
data_table2=rows2vars(data_table,'VariableNamingRule','preserve');
final=[clinical_data data_table2(:,2:end)];

%% Preprocess data

%Elimino le righe che hanno i counts nulli
geneData = table2array(data_table);
mask = geneData > 0;
sum_mask = sum(mask,2);
idx = sum_mask >= size(geneData,2)*80/100;
data_table(~idx,:) = [];


%Sostituisco ai vari stadi i valori 1,2,3 e 4
clinical=clinical_data(:,4);
clinical_array = convertvars(clinical,"ajcc_pathologic_stage",'string');
clinical_array=table2array(clinical_array);

for i=1:length(clinical_array)
    temp_str = clinical_array(i);
    switch temp_str
        case 'Stage I'
            temp_str=replace(temp_str,'Stage I','1');
        case 'Stage IA'
            temp_str = replace(temp_str,'Stage IA','1');
        case 'Stage IB'
            temp_str = replace(temp_str,'Stage IB','1');
        case 'Stage IC'
            temp_str = replace(temp_str,'Stage IC','1');

        case 'Stage II'
            temp_str= replace(temp_str,'Stage II','2');
        case 'Stage IIA'
            temp_str = replace(temp_str,'Stage IIA','2');
        case 'Stage IIB'
            temp_str = replace(temp_str,'Stage IIB','2');
        case 'Stage IIC'
            temp_str = replace(temp_str,'Stage IIC','2');

        case 'Stage III'
            temp_str= replace(temp_str,'Stage III','3');
        case 'Stage IIIA'
            temp_str = replace(temp_str,'Stage IIIA','3');
        case 'Stage IIIB'
            temp_str = replace(temp_str,'Stage IIIB','3');
        case 'Stage IIIC'
            temp_str = replace(temp_str,'Stage IIIC','3');

        case 'Stage IV'
            temp_str = replace(temp_str,'Stage IV','4');
        case 'Stage IVA'
            temp_str = replace(temp_str,'Stage IVA','4');
        case 'Stage IVB'
            temp_str = replace(temp_str,'Stage IVB','4');
        case 'Stage IVC'
            temp_str= replace (temp_str,'Stage IVC','4');

        case 'Stage X'
            temp_str=replace(temp_str,'Stage X','NaN');
    end
    clinical_array(i) = temp_str;
end

clinical_data.ajcc_pathologic_stage= clinical_array;

%righe NaN di patologic stage
labels_no_NaN=find(clinical_data.ajcc_pathologic_stage ~="NaN");
clinical_data= clinical_data(labels_no_NaN,:);
data_table= data_table(:,labels_no_NaN);



%% PLOT FEATURES CLINICHE
% PLOT ETA' PAZIENTI

eta=clinical_data(:,2);
eta=table2array(cell2table(eta.age));
figure
histogram(eta,'FaceColor','red');
title("Age distribution of patients")
xlabel("Age")
ylabel("Number of patients")


% PLOT RAZZA PAZIENTI
race=clinical_data(:,8);
lab_r1=find(race.race =="black or african american");
lab_r2=find(race.race =="white");
lab_r3=find(race.race =="asian");
lab_r4=find(race.race =="not reported");
lab_r5=find(race.race=="american indian or alaska native");

r1= categorical({'C1'});
r2= categorical({'C2'});
r3= categorical({'C3'});
r4= categorical({'C4'});
r5= categorical({'C5'});

l1=length(lab_r1);
l2=length(lab_r2);
l3=length(lab_r3);
l4=length(lab_r4);
l5=length(lab_r5);

bar(r1,l1);
hold on
bar(r2,l2);
bar(r3,l3);
bar(r4,l4);
bar(r5,l5);
title("Race of patients")
legend('Black or African American','White','Asian','Not Reported','American indian or Alaska native');
hold off

%  PLOT STADIO PAZIENTI
stadio=clinical_data(:,4);
stadio=table2array(stadio);
stadio= double(stadio);
figure
histogram(stadio,'FaceColor','blue')
title("Stage of patients")

% PLOT NUMERO PAZIENTI PER CLASSE
target= clinical_data(:,9);
%indici aggiornati 
labels_2=find(target.primary_diag =="Infiltrating duct carcinoma, NOS"); %indice dei pazienti con stadio 1
labels_1=find(target.primary_diag =="Lobular carcinoma, NOS"); %indice dei pazienti con stadio 2

figure
y=length(labels_2);
y2 = categorical({'Classe 1'});
bar(y2,y);
hold on
y3=length(labels_1);
y4 = categorical({'Classe 2'});
bar(y4,y3);
legend('Patients with Infiltrating duct carcinoma','Patients with Lobular carcinoma');
ylim([0 900])
title("Number of patients in each class")
hold off



%Sostituisco razza con numeri
clinical=clinical_data(:,8);
clinical_array = convertvars(clinical,"race",'string');
clinical_array=table2array(clinical_array);
for i=1:length(clinical_array)
    temp_str = clinical_array(i);
    switch temp_str
        case 'black or african american'
            temp_str=replace(temp_str,'black or african american','2');
       
        case 'white'
            temp_str=replace(temp_str,'white','1');
             
        case 'american indian or alaska native'
            temp_str=replace(temp_str,'american indian or alaska native','3');
        case 'asian'
            temp_str=replace(temp_str,'asian','4');
            
    end
    clinical_array(i) = temp_str;
end

clinical_data.race= clinical_array;
clinical_data = convertvars(clinical_data,"race",'double');

%elimino altre righe con valori NaN
missing_values = ismissing(clinical_data);
sum_missing_values = sum(missing_values, 2);
missing_rows = find(sum_missing_values > 0);
clinical_data(missing_rows, :) = [];
data_table(:, missing_rows) = [];
%target_feat(missing_rows, :) = [];

%target=clinical_data(:,8);    %output prima diagnosi 

labels_1=find(clinical_data.primary_diag =="Lobular carcinoma, NOS"); 
labels_2=find(clinical_data.primary_diag =="Infiltrating duct carcinoma, NOS"); 

labels12= sort([labels_1;labels_2]);

data_table= data_table(:,labels12);
clinical_data= clinical_data(labels12,:);

 %% salvo
writetable (data_table,'data_table.csv','WriteVariableNames',true,'WriteRowNames',true);
writetable (clinical_data,'data_table_clinical2.csv','WriteVariableNames',true,'WriteRowNames',true);

