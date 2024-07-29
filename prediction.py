from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import os

# Set the Spark home and update the PATH
os.environ['SPARK_HOME'] = '/home/ubuntu/spark-3.5.0-bin-hadoop3'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['SPARK_HOME'], 'bin')

# Initialize Spark session
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

def clean_column_names(df):
    new_column_names = []
    for c in df.columns:
        cleaned_name = c.strip().replace('"', '').replace("'", '').replace(';', '')
        new_column_names.append(cleaned_name)
    return df.toDF(*new_column_names)

# Load and clean training data
try:
    training_data = spark.read.csv("TrainingDataset.csv", header=True, inferSchema=True, sep=';')
    training_data = clean_column_names(training_data)
    training_data.show(5)  # Show the first few rows of the dataset for debugging
    training_data.printSchema()  # Print schema for debugging
    print(training_data.columns)  # Print column names for debugging
except Exception as e:
    print(f"Error loading training data: {e}")
    spark.stop()
    raise

# Ensure 'quality' column is present
if 'quality' not in training_data.columns:
    raise ValueError("The 'quality' column is missing from the training data.")
# Check for null values and handle them
for column in training_data.columns:
    null_count = training_data.filter(training_data[column].isNull()).count()
    if null_count > 0:
        print(f"Column {column} has {null_count} null values, filling with mean")
        mean_value = training_data.select(col(column)).agg({"*": "avg"}).collect()[0][0]
        training_data = training_data.na.fill({column: mean_value})

# Verify data types and cast if necessary
for column in training_data.columns:
    if column != 'quality':
        training_data = training_data.withColumn(column, col(column).cast("double"))

# Print cleaned and casted columns for debugging
print("Cleaned and casted columns:")
training_data.printSchema()
training_data.show(5)

# VectorAssembler to combine feature columns
feature_cols = [column for column in training_data.columns if column != 'quality']
print("Feature columns:", feature_cols)  # Print feature columns for debugging

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

try:
    assembled_data = assembler.transform(training_data)
    assembled_data.show(5)  # Show the first few rows of the assembled data
    assembled_data.printSchema()  # Print the schema to verify the 'features' column
except Exception as e:
    print(f"Error during VectorAssembler transformation: {e}")
    spark.stop()
    raise

# Evaluate the model on validation dataevaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
lr = LogisticRegression(labelCol="quality", featuresCol="features")lr_model = lr.fit(assembled_data)
predictions = lr_model.transform(assembled_data)
accuracy = evaluator.evaluate(predictions)

print(f"f1 Score: {accuracy}")
