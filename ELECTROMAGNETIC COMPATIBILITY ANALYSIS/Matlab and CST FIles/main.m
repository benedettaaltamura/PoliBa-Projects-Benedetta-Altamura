%%  

clear all
close all
clc

%% OTTIMIZZAZIONE LUNGHEZZA

% Directory in cui sono salvati i file (se diversa dalla cartella corrente)
directory = './'; % Cambia con il percorso della tua directory se necessario

% Ottieni tutti i file che iniziano con 'ottimizzazione' seguiti da un numero
filePattern = fullfile(directory, 'ottimizzazione*.txt');
fileList = dir(filePattern);

% Definisci una tavolozza di colori predefiniti (matrice di colori RGB)
colorOrder = lines(length(fileList));  % Usa la funzione 'lines' per colori distinti

% Creazione del grafico
figure;
hold on; % Mantiene tutti i grafici sullo stesso plot
grid on;
% Ciclo attraverso tutti i file trovati
for i = 1:length(fileList)
    % Legge il nome del file
    fileName = fullfile(directory, fileList(i).name);
    
    % Apri il file
    fid = fopen(fileName, 'r');
    
    % Leggi la prima riga (contiene i parametri)
    firstLine = fgetl(fid);
    
    % Estrai il valore di L dalla stringa della prima riga
    % Cerca il pattern 'L=valore'
    tokens = regexp(firstLine, 'L=([0-9]*\.?[0-9]+)', 'tokens');
    if ~isempty(tokens)
        L_value = tokens{1}{1}; % Il valore di L viene estratto come stringa
    else
        L_value = 'N/A'; % Se non trova L, mettiamo 'N/A'
    end
    
    % Ignora le prossime 2 righe
    for k = 1:2
        fgetl(fid); % Legge e ignora le prossime due righe
    end
    
    % Leggi i dati numerici dalla quarta riga in poi
    data = textscan(fid, '%f %f', 'CommentStyle', '#', 'Delimiter', ' ');
    
    % Chiudi il file
    fclose(fid);
    
    % Estrarre le colonne x e y
    x = data{1}; % Prima colonna: x
    y = data{2}; % Seconda colonna: y
    
    % Determina lo stile di linea e il colore
    if contains(fileList(i).name, 'ottimale')
        % Linea continua per il file 'lunghezza ottimale'
        plotStyle = '-'; % Linea continua
        color = 'r'; % Colore blu
        lineWidth = 2; % Spessore normale
    
    else
        % Linea tratteggiata e di colore diverso per tutti gli altri file
        plotStyle = '--'; % Linea tratteggiata
        color = colorOrder(i, :); % Assegna un colore diverso dalla tavolozza
        lineWidth = 1; % Spessore normale
    end
    
    % Plottiamo i dati con lo stile e il colore appropriato
    % Usa il valore di L per la legenda
    plot(x, y, plotStyle, 'Color', color, 'LineWidth', lineWidth, ...
        'DisplayName', ['L=', L_value, ' mm']); % 'DisplayName' per la legenda con L=valore
end

% Aggiunta di etichette e legenda
xlabel('Frequenza [GHz]');
ylabel('S_1_1');
title('Grafico delle ottimizzazioni');
xticks(0:2.5:25);
% Modifica della posizione della legenda e miglioramento della leggibilità
legend('show', 'Location', 'best', 'Interpreter', 'none'); % 'best' trova la posizione ottimale
hold off;

%% CAMPO DI RIFERIMENTO NEL VUOTO

% Creazione del grafico
figure;
hold on;

fid = fopen('E_RIFERIMENTO.txt', 'rt');
xE = [];
yE = [];
tline = fgets(fid);
        
while ischar(tline)
     data = strsplit(tline, '\t');
     xE(end+1) = str2double(data{1});
     yE(end+1) = str2double(data{2});
     tline = fgets(fid);
end

fclose(fid);

%Plot del singolo file
plot(xE, yE,'-' ,'LineWidth',1,'Color','r');


%Settaggi grafico e legenda
xlabel('Frequenza [GHz]');
ylabel('Campo Elettrico [dB*V/m]');
xticks(0:2.5:25);
title('Campo Elettrico di riferimento');
grid on;

%% CAMPO CON SCHERMATURA DI RAME

% Creazione del grafico
figure;
hold on;

fid = fopen('E_PROBE_TOTALE_RAME.txt', 'rt');
xE = [];
yE = [];
tline = fgets(fid);
        
while ischar(tline)
     data = strsplit(tline, '\t');
     xE(end+1) = str2double(data{1});
     yE(end+1) = str2double(data{2});
     tline = fgets(fid);
end
fclose(fid);

%Plot del singolo file
plot(xE, yE,'-' ,'LineWidth',1,'Color','r');


%Settaggi grafico e legenda
xlabel('Frequenza [GHz]');
ylabel('Campo Elettrico [dB*V/m]');
xticks(0:2.5:25);
grid on;

%% CAMPO CON APERTURA PARALLELA 

files = {
    'E_RAME_PARALLELA_lam2.txt',
    'E_RAME_PARALLELA_lam5.txt', %campo che soddisfa la condizione
    'E_RAME_PARALLELA_lam10.txt',
    'E_RAME_PARALLELA_lam15.txt',
    'E_RAME_PARALLELA_lam20.txt',
};

legend_names = {
    'La = 12 mm',
    'La = 4.8 mm',
    'La = 2.4 mm',
    'La = 1.6 mm',
    'La = 1.2 mm',
};

% Definisci gli stili delle linee per ciascun file
Widths = {0.8,2.5,0.8,0.8,0.8};
Colors = {"#0072BD",'r',"#EDB120", "#77AC30", "#7E2F8E"};

fileE0 = 'E_RIFERIMENTO.txt';

folderPath = './';

%Plot dei campi irradiati
plotEfield(folderPath,files,legend_names,Widths,Colors);
title('Campo elettrico irradiato - APERTURA PARALLELA');

%Plot dell'efficienza di schermatura
plotShieldingEffectivenessCombined(folderPath,fileE0, files,legend_names);
yline(60, 'k', '60 dB', 'LineWidth', 1.5,'HandleVisibility', 'off'); % Linea rossa tratteggiata
title('Apertura parallela');

%Plot dell'efficienza di schermatura alla frequenza di lavoro
files = {
    'E_RAME_PARALLELA_lam50.txt',
    'E_RAME_PARALLELA_lam20.txt',
    'E_RAME_PARALLELA_lam15.txt',
    'E_RAME_PARALLELA_lam10.txt',
    'E_RAME_PARALLELA_lam9.txt',
    'E_RAME_PARALLELA_lam8.txt',
    'E_RAME_PARALLELA_lam6.txt',
    'E_RAME_PARALLELA_lam5.txt',
    'E_RAME_PARALLELA_lam4.txt',
    'E_RAME_PARALLELA_lam2.txt',
};

legend_names = {
    'La = 0.48 mm',
    'La = 1.2 mm',
    'La = 1.6 mm',
    'La = 2.4 mm',
    'La = 2.67 mm',
    'La = 3 mm',
    'La = 4 mm',
    'La = 4.8 mm',
    'La = 6 mm',
    'La = 12 mm',
};

%Definisci distanze su asse x
x_ticks = [0.48, 1.2, 1.6, 2.4,2.67, 3, 4, 4.8, 6, 12];
plotWorkingFrequencyEffectiveness(folderPath, fileE0, files,legend_names,x_ticks);
yline(60, 'k', '60 dB', 'LineWidth', 1.5,'HandleVisibility', 'off'); % Linea rossa tratteggiata
title('SE a 12.5 GHz al variare di La - APERTURA PARALLELA');


%% CAMPO CON APERTURA NORMALE
% Campo schermato a diverse lambda
files = { 
    'E_RAME_NORMALE_lam2.txt',
    'E_RAME_NORMALE_lam4.txt',
    'E_RAME_NORMALE_lam6.txt',
    'E_RAME_NORMALE_lam10.txt', %campo che soddisfa la condizione
    'E_RAME_NORMALE_lam20.txt',
};

legend_names = {
    'La = 12 mm',
    'La = 6 mm',
    'La = 4 mm',
    'La = 2.4 mm',
    'La = 1.2 mm',
};

% Definisci gli stili delle linee per ciascun file
Widths = {0.8,0.8,0.8,3.5,0.8};
Colors = {"#0072BD", "#77AC30", "#EDB120", 'r', "#7E2F8E"};

fileE0 = 'E_RIFERIMENTO.txt';

folderPath = './';

%Plot dei campi irradiati
plotEfield(folderPath,files,legend_names,Widths,Colors);
title('Campo elettrico irradiato - APERTURA NOMRALE ');

%Plot dell'efficienza di schermatura
plotShieldingEffectivenessCombined(folderPath,fileE0, files,legend_names);
yline(60, 'k', '60 dB', 'LineWidth', 1.5,'HandleVisibility', 'off'); % Linea rossa tratteggiata
title('Apertura NORMALE');


files = { 
    'E_RAME_NORMALE_lam50.txt',
    'E_RAME_NORMALE_lam20.txt',
    'E_RAME_NORMALE_lam15.txt',
    'E_RAME_NORMALE_lam10.txt', %campo che soddisfa la condizione
    'E_RAME_NORMALE_lam9.txt',
    'E_RAME_NORMALE_lam8.txt',
    'E_RAME_NORMALE_lam6.txt',
    'E_RAME_NORMALE_lam5.txt',
    'E_RAME_NORMALE_lam4.txt',
    'E_RAME_NORMALE_lam2.txt',
};

legend_names = {
    'La = 0.48 mm',
    'La = 1.2 mm',
    'La = 1.6 mm',
    'La = 2.4 mm',
    'La = 2.6667 mm',
    'La = 3 mm',
    'La = 4 mm',
    'La = 4.8 mm',
    'La = 6 mm',
    'La = 12 mm',
};

%Definisci distanze su asse x
x_ticks = [0.48, 1.2, 1.6, 2.4,2.67, 3, 4, 4.8, 6, 12];
plotWorkingFrequencyEffectiveness(folderPath, fileE0, files,legend_names,x_ticks);
yline(60, 'k', '60 dB', 'LineWidth', 1.5,'HandleVisibility', 'off'); % Linea rossa tratteggiata
title('SE a 12.5 GHz al variare di La - APERTURA NORMALE');



%% TESSUTO OMOGENEO - MIDOLLO OSSEO

%file da plottare
files = { 
    'SAR_Midollo_Osseo_lineare_corrente1_maxmesh.txt',
    'SAR_Midollo_Osseo_lineare_corrente2_maxmesh.txt',
    'SAR_Midollo_Osseo_lineare_corrente3_maxmesh.txt',
};

legend_names = {
    '1*e-3 A',
    '1*e-2 A',
    '1*e-1 A',
};

% Plot del campo elettrico irradiato per ogni file
for i = 1:length(files)
    fid = fopen(files{i}, 'rt');
    xE = [];
    yE = [];
    tline = fgets(fid);
    while ischar(tline)
        data = strsplit(tline, '\t');
        xE(end+1) = str2double(data{1});
        yE(end+1) = str2double(data{2});
        tline = fgets(fid);
    end
    fclose(fid);
    
    %Plot del singolo file
    figure;
    plot(xE, yE,'-','LineWidth',2,'Color','r','DisplayName', legend_names{i});
    %Settaggi grafico e legenda
    xlabel('Spessore campione [mm]');
    ylabel('SAR [W/kg]');
    %legend('show','Location','best');
    grid on;
    title('SAR MIDOLLO OSSEO');

end

%% TESSUTO OMOGENEO - SCLERA/RETINA

%file da plottare
files = { 
    'SAR_Sclera_Retina_lineare_corrente1_maxmesh.txt',
    'SAR_Sclera_Retina_lineare_corrente2_maxmesh.txt',
    'SAR_Sclera_Retina_lineare_corrente3_maxmesh.txt',
};

legend_names = {
    '1*e-3 A',
    '1*e-2 A',
    '1*e-1 A',
};


% Plot del campo elettrico irradiato per ogni file
for i = 1:length(files)
    fid = fopen(files{i}, 'rt');
    xE = [];
    yE = [];
    tline = fgets(fid);
    while ischar(tline)
        data = strsplit(tline, '\t');
        xE(end+1) = str2double(data{1});
        yE(end+1) = str2double(data{2});
        tline = fgets(fid);
    end
    fclose(fid);
    
    %Plot del singolo file
    figure;
    plot(xE, yE,'-','LineWidth',2,'Color','r','DisplayName', legend_names{i});
    %Settaggi grafico e legenda
    xlabel('Spessore campione [mm]');
    ylabel('SAR [W/kg]');
    %legend('show','Location','best');
    grid on;
    title('SAR SCLERA/RETINA');

end

%% TESSUTO MULTISTRATO

%file 1 da plottare
file = 'SAR_multistrato_corrente1.txt';

fid = fopen(file, 'rt');
xE = [];
yE = [];
tline = fgets(fid);
while ischar(tline)
    data = strsplit(tline, '\t');
    xE(end+1) = str2double(data{1});
    yE(end+1) = str2double(data{2});
    tline = fgets(fid);
end
fclose(fid);

% Plot del singolo file
figure;
hold on

% Patch con legenda
p1 = patch([20.02 21 21 20.02],[0.00000 0.00000 0.06 0.06],[0.9 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Pelle');
p2 = patch([21 31 31 21],[0.0000 0.0000 0.06 0.06],[0.9804 0.9804 0.5294], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Grasso');
p3 = patch([31 36 36 31],[0.0000 0.0000 0.06 0.06],[0.8039 0.0784 0.0784], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Muscolo');
p4 = patch([36 44 44 36],[0.0000 0.0000 0.06 0.06],[0.9725 0.9725 0.9725], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Osso');

% Aggiungi la curva del file
plot(xE, yE, '-', 'LineWidth', 1, 'Color', [0 0 0], 'DisplayName', 'i = 1 mA');

% Settaggi grafico e legenda
xlabel('Spessore campione [mm]');
xlim([20 44]);
ylim([0 0.0006]);
ylabel('SAR [W/kg]');
legend('show', 'Location', 'northeastoutside'); % Posiziona la legenda fuori dal grafico
title('SAR Multistrato');
grid on;


%file 2 da plottare
file = 'SAR_multistrato_corrente2.txt';

fid = fopen(file, 'rt');
xE = [];
yE = [];
tline = fgets(fid);
while ischar(tline)
    data = strsplit(tline, '\t');
    xE(end+1) = str2double(data{1});
    yE(end+1) = str2double(data{2});
    tline = fgets(fid);
end
fclose(fid);

% Plot del singolo file
figure;
hold on

% Patch con legenda
p1 = patch([20.02 21 21 20.02],[0.000005 0.000005 0.06 0.06],[0.9 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Pelle');
p2 = patch([21 31 31 21],[0.00005 0.00005 0.06 0.06],[0.9804 0.9804 0.5294], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Grasso');
p3 = patch([31 36 36 31],[0.00005 0.00005 0.06 0.06],[0.8039 0.0784 0.0784], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Muscolo');
p4 = patch([36 44 44 36],[0.00005 0.00005 0.06 0.06],[0.9725 0.9725 0.9725], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Osso');

% Aggiungi la curva del file
plot(xE, yE, '-', 'LineWidth', 1, 'Color', [0 0 0], 'DisplayName', 'i = 10 mA');

% Settaggi grafico e legenda
xlabel('Spessore campione [mm]');
xlim([20 44]);
ylim([0 0.06]);
ylabel('SAR [W/kg]');
legend('show', 'Location', 'northeastoutside'); % Posiziona la legenda fuori dal grafico
title('SAR Multistrato');
grid on;


%file 3 da plottare
file = 'SAR_multistrato_corrente3.txt';
i = 1;
fid = fopen(file, 'rt');
xE = [];
yE = [];
tline = fgets(fid);
while ischar(tline)
    data = strsplit(tline, '\t');
    xE(end+1) = str2double(data{1});
    yE(end+1) = str2double(data{2});
    tline = fgets(fid);
end
fclose(fid);

% Plot del singolo file
figure;
hold on

% Patch con legenda
p1 = patch([20.02 21 21 20.02],[0.000005 0.000005 6 6],[0.9 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Pelle');
p2 = patch([21 31 31 21],[0.00005 0.00005 6 6],[0.9804 0.9804 0.5294], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Grasso');
p3 = patch([31 36 36 31],[0.00005 0.00005 6 6],[0.8039 0.0784 0.0784], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Muscolo');
p4 = patch([36 44 44 36],[0.00005 0.00005 6 6],[0.9725 0.9725 0.9725], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Osso');

% Aggiungi la curva del file
plot(xE, yE, '-', 'LineWidth', 1, 'Color', [0 0 0], 'DisplayName', 'i = 100 mA');

% Settaggi grafico e legenda
xlabel('Spessore campione [mm]');
xlim([20 44]);
ylim([0 6]);
ylabel('SAR [W/kg]');
legend('show', 'Location', 'northeastoutside'); % Posiziona la legenda fuori dal grafico
title('SAR Multistrato');
grid on;

%% CONFRONTO TESSUTO MULTISTRATO CON E SENZA IMPIANTO

%file 2 da plottare
file = 'SAR_multistrato_corrente2.txt';
file2= 'SAR_impianto_corrente2.txt';

fid = fopen(file, 'rt');
xE = [];
yE = [];
tline = fgets(fid);
while ischar(tline)
    data = strsplit(tline, '\t');
    xE(end+1) = str2double(data{1});
    yE(end+1) = str2double(data{2});
    tline = fgets(fid);
end
fclose(fid);

fid2 = fopen(file2, 'rt');
xE2 = [];
yE2 = [];
tline2 = fgets(fid2);
while ischar(tline2)
    data2 = strsplit(tline2, '\t');
    xE2(end+1) = str2double(data2{1});
    yE2(end+1) = str2double(data2{2});
    tline2 = fgets(fid2);
end
fclose(fid2);

% Plot del singolo file
figure;
hold on

% Patch con legenda
p1 = patch([20.02 21 21 20.02],[0.000005 0.000005 0.06 0.06],[0.9 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Pelle');
p2 = patch([21 31 31 21],[0.00005 0.00005 0.06 0.06],[0.9804 0.9804 0.5294], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Grasso');
p3 = patch([31 36 36 31],[0.00005 0.00005 0.06 0.06],[0.8039 0.0784 0.0784], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Muscolo');
p4 = patch([36 44 44 36],[0.00005 0.00005 0.06 0.06],[0.9725 0.9725 0.9725], 'EdgeColor', 'none', 'FaceAlpha', 0.5, 'DisplayName', 'Osso');

% Aggiungi la curva del file
plot(xE, yE, '-', 'LineWidth', 1, 'Color', [0 0 0], 'DisplayName', 'Senza impianto');
plot(xE2, yE2, '-', 'LineWidth', 1, 'Color', [0 0 1], 'DisplayName', 'Con impianto');
% Settaggi grafico e legenda
xlabel('Spessore campione [mm]');
xlim([20 44]);
ylim([0 0.06]);
ylabel('SAR [W/kg]');
legend('show', 'Location', 'northeastoutside'); % Posiziona la legenda fuori dal grafico
title('SAR Multistrato');
grid on;

