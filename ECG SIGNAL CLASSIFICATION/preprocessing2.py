import pandas as pd
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from matplotlib import pyplot as plt

def undersample_majority_classes(df, target_col, seed=42):
    minority_class_count = df.groupBy(target_col).count().agg({"count": "min"}).collect()[0][0]
    print("Minority class records: ")
    print(minority_class_count)
    classes = df.select(target_col).distinct().collect()
    sampled_df = None

    for cls in classes:
        cls_df = df.filter(F.col(target_col) == cls[target_col])
        fraction = minority_class_count / cls_df.count()
        undersampled_cls_df = cls_df.sample(withReplacement=False, fraction=fraction, seed=seed)
        sampled_df = undersampled_cls_df if sampled_df is None else sampled_df.union(undersampled_cls_df)

    return sampled_df

def preprocessing(train_df, test_df):
    train_df.persist(StorageLevel.MEMORY_AND_DISK)
    train_df.count()

    test_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df.count()

    # Identify the target column (last column)
    train_last_column_name = train_df.columns[-1]

    # Undersample majority classes
    train_df_undersampled = undersample_majority_classes(train_df, train_last_column_name)
    print(f'Undersampled training set count: {train_df_undersampled.count()}')

    feature_cols = train_df_undersampled.columns[:-1]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="featuresAssembled")
    scaler = StandardScaler(inputCol="featuresAssembled", outputCol="scaledFeatures", withStd=True, withMean=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    pipeline_model = pipeline.fit(train_df_undersampled)
    train_df_scaled = pipeline_model.transform(train_df_undersampled)
    test_df_scaled = pipeline_model.transform(test_df)

    # PCA iniziale per calcolare il numero di componenti
    pca_full = PCA(k=len(feature_cols), inputCol="scaledFeatures", outputCol="pcaFeaturesFull")
    pca_model_full = pca_full.fit(train_df_scaled)

    # Numero di componenti che spiegano il 95% della varianza
    explained_variance = pca_model_full.explainedVariance.toArray()
    explained_variance_cumsum = pd.Series(explained_variance).cumsum()
    n_components = (explained_variance_cumsum >= 0.95).idxmax() + 1
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

    # PCA con il numero corretto di componenti
    pca = PCA(k=n_components, inputCol="scaledFeatures", outputCol="pcaFeatures")
    pca_model = pca.fit(train_df_scaled)
    train_pca = pca_model.transform(train_df_scaled).select("pcaFeatures", train_last_column_name)
    test_pca = pca_model.transform(test_df_scaled).select("pcaFeatures", train_last_column_name)

    train_df.unpersist()
    test_df.unpersist()
    return train_pca, test_pca
