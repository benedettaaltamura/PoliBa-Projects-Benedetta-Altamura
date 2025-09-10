import os
import pandas as pd
import time


def split_csv(input_file, output_folder, chunk_size=100, delay=5):
    # Leggi il file CSV originale
    df = pd.read_csv(input_file)

    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Ottieni il numero totale di righe
    total_rows = len(df)

    # Calcola il numero di file che verranno creati
    num_files = (total_rows // chunk_size) + (1 if total_rows % chunk_size != 0 else 0)

    for i in range(num_files):
        start_row = i * chunk_size
        end_row = min((i + 1) * chunk_size, total_rows)

        # Estrai il chunk di righe dal dataframe
        chunk_df = df.iloc[start_row:end_row]

        # Crea il nome del file di output
        output_file = os.path.join(output_folder, f'output_{i + 1}.csv')

        # Scrivi il chunk nel file CSV
        chunk_df.to_csv(output_file, index=False)
        print(f'File creato: {output_file}')

        # Attendi per il tempo specificato prima di creare il prossimo file
        if end_row < total_rows:
            time.sleep(delay)


input_file = "/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/Dataset/mitbih_test.csv"
output_dir = '/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/Dataset/csv_streaming'
split_csv(input_file, output_dir)
