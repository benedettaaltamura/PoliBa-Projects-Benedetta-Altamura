% Funzione per il plotting dei dati su uno stesso grafico
function plotShieldingEffectivenessCombined(folderPath, fileE0, files,names)

    % Lettura dei dati dal campo di riferimento nel vuoto
    fid = fopen(fullfile(folderPath, fileE0), 'rt');
    xE0 = [];
    yE0 = [];
    tline = fgets(fid);
   
    while ischar(tline)
        data = strsplit(tline, '\t');
        xE0(end+1) = str2double(data{1});
        yE0(end+1) = str2double(data{2});
        tline = fgets(fid);
    end
    fclose(fid);

    % Creazione del grafico
    figure;
    hold on;
    grid on;
    %Condizione per il superamento dei 60 dB
    Condition = false; % quando grafico di SE supera i 60 dB,viene modificata in true 

    % Plot dell'efficienza di schermatura per ogni file
    for i = 1:length(files)
        fid = fopen(fullfile(folderPath, files{i}), 'rt');
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

        % Calcolo della formula
        result = yE0 - yE;
        
         % Verifica se la condizione è soddisfatta
        if (Condition == false)
            if (result >= 60) 
                % Plot dei dati
                plot(xE, result, '-','LineWidth',1.5,'Color','r','DisplayName', names{i});
                Condition = true;  % Imposta la variabile a true dopo il primo plot
            else
                plot(xE, result, '--','LineWidth',1,'DisplayName', names{i});
            end
        else
            plot(xE, result, '--','LineWidth',1,'DisplayName', names{i});
        end
    end
    
    %Settaggi grafico e legenda
    xlabel('Frequenze [GHz]');
    ylabel('Efficienza di schermatura [dB]');
    legend('show','Location','bestoutside');

    % Aggiunta della linea a y=60
    %yline(60, 'b--', '60 dB', 'LineWidth', 2); % Linea nera tratteggiata
    % Imposta i limiti degli assi x e y
    xlim([0 25]);
    ylim([0 120]);
    hold off
    % Imposta i valori sull'asse x ogni 2,5 unità
    xticks(0:2.5:25);
    yticks(0:20:120);
    
end




