import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel

# Inizializza la sessione Spark
spark = SparkSession.builder.appName("RealtimeCSVAnalysis").getOrCreate()

# Definisci lo schema del file CSV in base al numero di colonne del dataset
num_columns = 188  # Specifica il numero esatto di colonne del tuo dataset
schema = StructType([StructField(f"_c{i}", FloatType(), True) for i in range(num_columns)])

# Percorso della cartella dei file CSV
csv_folder_path = "/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/Dataset/csv_streaming"

# Carica il modello di preprocessing completo (PipelineModel che include VectorAssembler, StandardScaler e PCA)
preprocessing_pipeline = PipelineModel.load("/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/pipeline_model")

# Carica il modello di predizione (CrossValidatorModel)
prediction_model = CrossValidatorModel.load("/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/ann_model")

# Leggi i file CSV come uno stream con lo schema specificato
csv_stream = (
    spark.readStream
    .option("sep", ",")
    .option("header", "false")
    .schema(schema)
    .csv(csv_folder_path)
)

# Rimuovi o sostituisci i valori null nei dati
csv_stream = csv_stream.na.fill(0)  # Sostituisci i valori null con 0

# Applica il modello di preprocessing ai dati in streaming
preprocessed_data = preprocessing_pipeline.transform(csv_stream)

# Applica il modello di predizione ai dati preprocessati in streaming
predictions = prediction_model.bestModel.transform(preprocessed_data)

# Seleziona la colonna di interesse
predictions = predictions.select("pcaFeatures", "prediction")

# Scrivi il risultato nello stream di output (console)
query = (
    predictions.writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", "false")  # Opzione per evitare il troncamento dell'output in console
    .start()
)

# Avvia lo stream
query.awaitTermination()
