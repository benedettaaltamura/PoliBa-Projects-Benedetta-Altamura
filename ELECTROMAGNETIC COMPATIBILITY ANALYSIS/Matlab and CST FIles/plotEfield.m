function plotEfield(folderPath, files,names,lineWidths, lineColors)
    
    % Creazione del grafico
    figure;
    hold on;

    % Plot del campo elettrico irradiato per ogni file
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
        
        %Plot del singolo file
        plot(xE, yE,'-','LineWidth',lineWidths{i},'Color',lineColors{i},'DisplayName', names{i});
    
    end

    %Settaggi grafico e legenda
    xlabel('Frequenze [GHz]');
    ylabel('Campo Elettrico Irradiato [dB*V/m]');
    legend('show','Location','bestoutside');
    xticks(0:2.5:25);
    grid on;

end