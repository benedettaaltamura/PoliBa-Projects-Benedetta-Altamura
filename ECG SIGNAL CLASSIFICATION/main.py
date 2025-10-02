import numpy as np
from matplotlib import pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum
from preprocessing import preprocessing
import seaborn as sns
import time
from model_training_rf import model_training_rf
from model_training_ann2 import model_training_ann

start_time = time.time()

if __name__ == '__main__':
    spark = SparkSession.builder \
        .appName("ECG Analyst") \
        .config("spark.driver.memory", "7g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.driver.maxResultsize", "4g") \
        .config("spark.sql.debug.maxToStringFields", "1000") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Importazione file csv e creazione Dataframe Spark
    train_path = "/home/serena/BigData/Spark/pythonProject1/Dataset/mitbih_train.csv"
    test_path = "/home/serena/BigData/Spark/pythonProject1/Dataset/mitbih_test.csv"
    train_df = spark.read.csv(train_path, header=False, inferSchema=True)
    test_df = spark.read.csv(test_path, header=False, inferSchema=True)

    # Controllo per valori mancanti
    missing_values_train = train_df.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in train_df.columns]
    ).collect()
    missing_values_train_dict = {col: missing_values_train[0][col] for col in train_df.columns}
    total_missing_train = sum(missing_values_train_dict.values())
    print("Valori mancanti nei dati di addestramento:")
    print(missing_values_train_dict)
    print(f"Totale valori mancanti nei dati di addestramento: {total_missing_train}")

    missing_values_test = test_df.select(
        [spark_sum(col(c).isNull().cast("int")).alias(c) for c in test_df.columns]
    ).collect()
    missing_values_test_dict = {col: missing_values_test[0][col] for col in test_df.columns}
    total_missing_test = sum(missing_values_test_dict.values())
    print("Valori mancanti nei dati di test:")
    print(missing_values_test_dict)
    print(f"Totale valori mancanti nei dati di test: {total_missing_test}")

    columns = train_df.columns
    renamed_columns = [str(i) if i < len(columns) - 1 else columns[i] for i in range(len(columns))]

    # for i in range(len(columns) - 1):
    # train_df = train_df.withColumnRenamed(columns[i], renamed_columns[i])
    # test_df = test_df.withColumnRenamed(columns[i], renamed_columns[i])

    train_df.show(5)
    # train_df.printSchema()

    print("Numero di record (train):")
    train_df.groupBy(renamed_columns[-1]).count().show()

    print("Numero di record (test):")
    test_df.groupBy(renamed_columns[-1]).count().show()

    classes = {0: 'Battito normale',
               1: 'Battito prematuro sopraventricolare',
               2: 'Contrazione ventricolare prematura',
               3: 'Fusione di battito ventricolare e normale',
               4: 'Battito non classificabile'}

    ## Esplorazione Dataset
    train_pd = train_df.toPandas()
    test_pd = test_df.toPandas()

    # Distribuzione classi nel dataset di training
    plt.figure(figsize=(10, 6))
    sns.countplot(x=train_pd.iloc[:, -1])
    plt.title('Distribuzione delle classi nei dati di addestramento')
    plt.xlabel('Etichetta di Classe')
    plt.ylabel('Conteggio')
    plt.xticks(ticks=np.arange(len(classes)), labels=[classes[label] for label in range(len(classes))], rotation=45)
    # plt.show()
    plt.savefig('Distribuzione_classi_dati_addestramento.png')
    plt.close()
    print("\"Distribuzione_classi_dati_addestramento.png\" salvata.")

    # Visualizzazione alcuni segnali ECG
    def plot_ecg_signals(data, labels, class_name, n_samples=5):
        plt.figure(figsize=(15, 10))
        for i in range(n_samples):
            plt.subplot(n_samples, 1, i + 1)
            plt.plot(data.iloc[i, :-1])
            plt.title(f"Segnale ECG - {class_name}")
            plt.xlabel('Tempo')
            plt.ylabel('Ampiezza')
            ticks = np.arange(0, data.shape[1] - 1, step=50)
            plt.xticks(ticks, labels=[str(tick) for tick in ticks])  # Imposta i tick dell'asse x a intervalli di 50
        plt.tight_layout()
        fig_name = class_name.replace(" ", "_")
        plt.savefig(f'{fig_name}.png')  # Salva l'immagine con il nome della classe
        print(f"\"{fig_name}.png\" salvata.")
        plt.close()

    for label in classes.keys():
        sample_data = train_pd[train_pd.iloc[:, -1] == label]
        plot_ecg_signals(sample_data, sample_data.iloc[:, -1].values, classes[label], n_samples=5)

    end_time_1 = time.time()
    #test_df.show(3)
    # Pre-processing
    # train_df_final, test_df_final = preprocessing2(train_df, test_df)
    train_df_final, test_df_final = preprocessing(train_pd, test_df, spark)

    # train_df_final.printSchema()
    # train_df_final.show(3)
    # test_df_final.printSchema()
    # test_df_final.show(3)
    end_time_2 = time.time()
    print(f"Tempo di esecuzione del pre-processing: {end_time_2 - start_time:.2f} secondi")

    # Modello ANN
    model_training_ann(train_df_final, test_df_final, classes)
    end_time_3 = time.time()
    print(f"Tempo di esecuzione modello ANN: {end_time_3 - end_time_2:.2f} secondi")

    #Modello Random Forest
    #model_training_rf(train_df_final, test_df_final, classes)
    #end_time_4 = time.time()
    #print(f"Tempo di esecuzione modello Random Forest: {end_time_4 - end_time_2:.2f} secondi")

    end_time_5 = time.time()
    print(f"Tempo di esecuzione complessivo: {end_time_5 - start_time:.2f} secondi")

    # Save execution times to a text file
    with open("execution_times_ann.txt", "w") as file:
        file.write(f"")
        file.write(f"Pre-processing execution time: {end_time_2 - start_time:.2f} seconds\n")
        file.write(f"ANN model execution time: {end_time_3 - end_time_2:.2f} seconds\n")
        file.write(f"Total execution time: {end_time_5 - start_time:.2f} seconds\n")
    print("\"execution_times_ann.txt\" saved.")
