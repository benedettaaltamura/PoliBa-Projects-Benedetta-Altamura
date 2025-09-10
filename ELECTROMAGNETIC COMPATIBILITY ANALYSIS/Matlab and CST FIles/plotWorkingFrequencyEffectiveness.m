% Funzione per il plotting dei dati su uno stesso grafico
function plotWorkingFrequencyEffectiveness(folderPath, fileE0, files,names,x_ticks)

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

    %Creazione vettori vuoti per contenere coordinate grafico
    x_values = [];
    y_values = [];
    
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

        % Trova l'indice corrispondente a xE = 12.5
        [~, idx] = min(abs(xE - 12.5));

        % Seleziona solo il valore di result associato a xE = 12.5
        result_12_5 = yE0(idx) - yE(idx);

        % Plot dei dati
        %plot(i, result_12_5, '-o', 'LineWidth', 2);

        % Aggiorna gli array di output
        x_values(end+1) = i;
        y_values(end+1) = result_12_5;

    end
    
    %Per mostrare i valori su asse x in ordine crescente
    % y_values = flip(y_values); 
    % x_values = flip(x_values);

    % Interpolazione dei valori su un insieme uniforme di punti
    x_uniform = x_values;
    y_interpolated = interp1(x_values, y_values, x_uniform, 'spline');
    
    % Creazione del grafico per la seconda figura sovrapposta alla prima
    figure;
    scatter(x_ticks, y_values,'o','LineWidth',2, 'DisplayName', 'Dati Simulati');
    hold on;
    plot(x_ticks,y_interpolated,"LineWidth",1.5,"Color","red","DisplayName","Curva Interpolata");
    xlabel('Lunghezze [mm]');
    ylabel('Efficienza di schermatura [dB]');
    legend('show','Location','bestoutside');
    grid on;
    ylim([0 100]);
    xticks(x_ticks);
    yticks(0:10:100);
    
end




