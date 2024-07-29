from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
import os

# Set the Spark home and update the PATH
os.environ['SPARK_HOME'] = '/home/ubuntu/spark-3.5.0-bin-hadoop3'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['SPARK_HOME'], 'bin')

# Initialize Spark session with S3 configurations
spark = SparkSession.builder \
    .appName("WineQualityPrediction") \
    .config("spark.hadoop.fs.s3a.access.key", "<YOUR_AWS_ACCESS_KEY>") \
    .config("spark.hadoop.fs.s3a.secret.key", "<YOUR_AWS_SECRET_KEY>") \
    .config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com") \
    .getOrCreate()

def clean_column_names(df):
    new_column_names = []
    for c in df.columns:
        cleaned_name = c.strip().replace('"', '').replace("'", '').replace(';', '')
        new_column_names.append(cleaned_name)
    return df.toDF(*new_column_names)

# Load and clean training data from S3
s3_path = "s3a://your-bucket-name/TrainingDataset.csv"
try:
    training_data = spark.read.csv(s3_path, header=True, inferSchema=True, sep=';')
    training_data = clean_column_names(training_data)
except Exception as e:
    print(f"Error loading training data from S3: {e}")
    spark.stop()
    raise

# Ensure 'quality' column is present
if 'quality' not in training_data.columns:
    raise ValueError("The 'quality' column is missing from the training data.")

# Check for null values and handle them
for column in training_data.columns:
    null_count = training_data.filter(training_data[column].isNull()).count()
    if null_count > 0:
        mean_value = training_data.select(col(column)).agg({"*": "avg"}).collect()[0][0]
        training_data = training_data.na.fill({column: mean_value})

# Verify data types and cast if necessary
for column in training_data.columns:
    if column != 'quality':
        training_data = training_data.withColumn(column, col(column).cast("double"))

# VectorAssembler to combine feature columns
feature_cols = [column for column in training_data.columns if column != 'quality']
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

assembled_data = assembler.transform(training_data)

# Split data into training and validation sets
(training_data, validation_data) = assembled_data.randomSplit([0.7, 0.3])

# Train Logistic Regression model
lr = LogisticRegression(labelCol="quality", featuresCol="features")
lr_model = lr.fit(training_data)

# Save the trained model locally
local_model_path = "/home/ubuntu/logistic_regression_model"
lr_model.save(local_model_path)

# Validate the model
predictions = lr_model.transform(validation_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"Model accuracy: {accuracy}")

spark.stop()

