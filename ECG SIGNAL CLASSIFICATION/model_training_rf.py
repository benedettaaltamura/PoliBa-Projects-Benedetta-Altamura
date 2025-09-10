from pyspark import StorageLevel
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def model_training_rf(train_df_pca, test_df_pca, classes):
    train_df_pca.repartition(100)
    test_df_pca.repartition(100)
    train_df_pca.persist(StorageLevel.MEMORY_AND_DISK)
    test_df_pca.persist(StorageLevel.MEMORY_AND_DISK)

    # Inizializzazione RandomForestClassifier
    rf = RandomForestClassifier(labelCol="class",
                                featuresCol="pcaFeatures",
                                numTrees=100,
                                seed=42,
                                minInstancesPerNode=2,
                                maxDepth=11)

    # Creazione griglia di parametri per la cross-validazione
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [50, 100])
                 .addGrid(rf.maxDepth, [5, 11])
                 .build())


    evaluator = MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction", metricName="accuracy")
    cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    print("Model training with cross-validation...")
    cv_model = cv.fit(train_df_pca)
    best_model = cv_model.bestModel
    # Salva il modello addestrato
    best_model.write().overwrite().save('/home/serena/Documenti/Progetto BigDataAnalytics/pythonProject_ECG/rf_model')

    # Make predictions
    print("Inferencing...")
    predictions = best_model.transform(test_df_pca)

    # Calcolo metriche per ciascuna classe
    prediction_and_labels = predictions.select("prediction", "class").rdd
    metrics = MulticlassMetrics(prediction_and_labels)

    for label in classes.keys():
        class_accuracy = metrics.accuracy
        class_precision = metrics.precision(float(label))
        class_recall = metrics.recall(float(label))
        class_f1 = metrics.fMeasure(float(label))
        print(f"\nMetrics for class {classes[label]} (label {label}):")
        print(f" Accuracy: {class_accuracy:.4f}")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  F1 Score: {class_f1:.4f}")

    # Matrice di confusione
    confusion_matrix = metrics.confusionMatrix().toArray()

    # Conversione in percentuali
    confusion_matrix_percent = (confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]) * 100

    print("\nConfusion Matrix (Percentages):")
    print(confusion_matrix_percent)

    # Grafico matrice di confusione
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_percent, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=[classes[i] for i in classes], yticklabels=[classes[i] for i in classes])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.title("Confusion Matrix (Percentages)")
    plt.savefig('confusion_matrix_rf_cv.png')  # Save the plot as an image
    plt.close()

    # Implementazione curve ROC
    print("ROC Curve ...")
    y_test = np.array(predictions.select("class").collect())
    y_score = np.array(predictions.select("probability").collect()).squeeze()

    # Binarizzazione delle labels
    y_test_binarized = label_binarize(y_test, classes=list(classes.keys()))

    # One-vs-Rest ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Grafico One-vs-Rest ROC
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='blue', lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Class {classes[i]}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_class_{i}_rf.png')  # Save the plot as an image
        plt.close()

    train_df_pca.unpersist()
    test_df_pca.unpersist()
