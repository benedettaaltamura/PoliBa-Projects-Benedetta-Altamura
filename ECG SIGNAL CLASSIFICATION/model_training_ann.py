from pyspark import StorageLevel
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.mllib.evaluation import MulticlassMetrics
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def model_training_ann(train_df_pca, test_df_pca, classes):

    train_df_pca.repartition(200)
    test_df_pca.repartition(200)
    train_df_pca.persist(StorageLevel.MEMORY_AND_DISK)
    test_df_pca.persist(StorageLevel.MEMORY_AND_DISK)

    # Define the layers of the neural network
    input_layer_size = train_df_pca.select("pcaFeatures").first()[0].size
    layers1 = [input_layer_size, 70, 20, len(classes)]
    layers2 = [input_layer_size, 10, 5, len(classes)]

    # Initialize the MultilayerPerceptronClassifier
    mlp = MultilayerPerceptronClassifier(labelCol="class", featuresCol="pcaFeatures", maxIter=50, blockSize=64,
                                         seed=1234)

    # Create a ParamGridBuilder for hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(mlp.layers, [layers1, layers2]) \
        .addGrid(mlp.maxIter, [10, 450]) \
        .build()

    # Create a CrossValidator
    crossval = CrossValidator(estimator=mlp,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="class", predictionCol="prediction", metricName="accuracy"),
                              numFolds=3)

    # Train the model using cross-validation
    print("Model training ...")
    cvModel = crossval.fit(train_df_pca)
    best_model = cvModel.bestModel

    # Make predictions
    print("Inferencing ... ")
    predictions = best_model.transform(test_df_pca)
    #results = predictions.toPandas()
    #results.to_excel("results_ann.xlsx")

    # Select example rows to display
    predictions.select("class", "prediction", "probability").show(5)

    # Calculate metrics for each class
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

    # Confusion matrix
    confusion_matrix = metrics.confusionMatrix().toArray()

    # Convert to percentages
    confusion_matrix_percent = (confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]) * 100

    print("\nConfusion Matrix (Percentages):")
    print(confusion_matrix_percent)

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_percent, annot=True, fmt=".1f", cmap="YlGnBu", xticklabels=[classes[i] for i in classes], yticklabels=[classes[i] for i in classes])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.title("Confusion Matrix (Percentages)")
    plt.savefig('confusion_matrix_nn.png')  # Save the plot as an image
    plt.close()

    # ROC Curve Implementation
    print("ROC Curve ...")
    y_test = np.array(predictions.select("class").collect())
    y_score = np.array(predictions.select("probability").collect()).squeeze()

    # Binarize the labels
    y_test_binarized = label_binarize(y_test, classes=list(classes.keys()))

    # One-vs-Rest ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plotting One-vs-Rest ROC
        plt.figure()
        plt.plot(fpr[i], tpr[i], color='blue', lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - Class {classes[i]}')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_class_{i}.png')  # Save the plot as an image
        plt.close()

    train_df_pca.unpersist()
    test_df_pca.unpersist()




