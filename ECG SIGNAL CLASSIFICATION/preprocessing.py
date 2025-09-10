import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
import seaborn as sns
from pyspark.sql.types import StructType, StructField, DoubleType

def preprocessing(train_pd, test_df, spark=None):

    # Split Dataframes Pandas
    X_train = train_pd.iloc[:, :-1]
    y_train = train_pd.iloc[:, -1]

    # Resampling dei dati
    print("SMOTE oversampling ...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    resampled_distribution = y_train_res.value_counts()
    print("Resampled class distribution: ")
    print(resampled_distribution)

    # Visualizzazione della distribuzione dopo il resampling
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train_res)
    plt.title('Resampled Class Distribution in Training Data')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    #plt.show()
    plt.savefig("Class Distribution after smote.png")
    plt.close()
    print(" \"Class Distribution after smote.png\" saved.")

    # Convert resampled data back to Spark DataFrame
    resampled_pd = pd.concat([X_train_res, y_train_res], axis=1)
    resampled_pd.columns = list(X_train.columns) + ['label']  # Ensure the label column is named correctly
    schema = StructType([StructField(col, DoubleType(), True) for col in X_train.columns] +
                        [StructField('label', DoubleType(), True)])

    train_df_resampled = spark.createDataFrame(resampled_pd, schema)
    train_last_column_name = train_df_resampled.columns[-1]
    train_df_resampled = train_df_resampled.withColumnRenamed(train_last_column_name, "class").repartition(200)
    train_df_resampled.persist(StorageLevel.MEMORY_AND_DISK)
    print("Number of records (train_df_resampled): ")
    print(train_df_resampled.count())

    test_last_column_name = test_df.columns[-1]
    test_df = test_df.withColumnRenamed(test_last_column_name, "class").repartition(200)
    test_df.persist(StorageLevel.MEMORY_AND_DISK)
    print("Number of records (test_df): ")
    print(test_df.count())

    feature_cols = train_df_resampled.columns[:-1]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="featuresAssembled")
    scaler = StandardScaler(inputCol="featuresAssembled", outputCol="scaledFeatures", withStd=True, withMean=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = pipeline.fit(train_df_resampled)
    train_df_scaled = pipeline_model.transform(train_df_resampled)
    test_df_scaled = pipeline_model.transform(test_df)

    # PCA iniziale per calcolare il numero di componenti
    pca_full = PCA(k=len(feature_cols), inputCol="scaledFeatures", outputCol="pcaFeaturesFull")
    pca_model_full = pca_full.fit(train_df_scaled)

    # Numero di componenti che spiegano il 95% della varianza
    explained_variance = pca_model_full.explainedVariance.toArray()
    explained_variance_cumsum = pd.Series(explained_variance).cumsum()
    n_components = (explained_variance_cumsum >= 0.95).idxmax()+1
    print(f'Number of components explaining 95% of variance: {n_components}')

    # Plot the explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance_cumsum, marker='o', linestyle='--')
    plt.title('Explained Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid()
    plt.savefig("Explained Variance by Components.png")
    plt.close()
    print(" \"Explained Variance by Components.png\" saved.")
    #plt.show()


    # PCA con il numero corretto di componenti
    pca = PCA(k=n_components, inputCol="scaledFeatures", outputCol="pcaFeatures")
    pca_model = pca.fit(train_df_scaled)
    train_pca = pca_model.transform(train_df_scaled).select("pcaFeatures", "class")
    test_pca = pca_model.transform(test_df_scaled).select("pcaFeatures", "class")

    #train_df_resampled.show(3)
    #test_pca.show(3)

    train_df_resampled.unpersist()
    test_df.unpersist()
    return train_pca, test_pca






