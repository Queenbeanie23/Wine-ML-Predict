
# Wine-ML-Predict

## Overview
This project aims to develop a wine quality prediction machine learning model using Apache Spark on the Amazon AWS cloud platform. The model is trained in parallel on multiple EC2 instances and is packaged in a Docker container to facilitate deployment across various environments.

## Goals
- Develop a parallel ML application using Apache Spark.
- Utilize Spark’s MLlib to create and use an ML model.
- Deploy the ML model using Docker for simplified deployment.

## Project Structure
- **Training Dataset:** `TrainingDataset.csv` 
- **Validation Dataset:** `ValidationDataset.csv` 

## Output
The application will output a measure of prediction performance, specifically the F1 score.

## Model Implementation
1. **Training Model:** Use Spark MLlib for training a wine quality prediction model using the training dataset.
2. **Validation:** Use the validation dataset to check and optimize model performance.
3. **Prediction:** Implement a prediction application that uses the trained model to predict wine quality.
4. **Docker:** Package the prediction application in a Docker container for easy deployment.

## Requirements
- Java on Ubuntu Linux
- Apache Spark
- Docker
- AWS account with access to EC2 instances

## Setup Instructions

### Step 1: Setting Up the Cloud Environment

#### AWS Setup
1. Launch 4 EC2 instances for model training.
2. Launch 1 EC2 instance for the prediction application.

#### Install Java and Spark
On each EC2 instance, install Java and Apache Spark:
```sh
sudo apt-get update
sudo apt-get install openjdk-11-jdk
wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz
tar xvf spark-3.1.2-bin-hadoop3.2.tgz
sudo mv spark-3.1.2-bin-hadoop3.2 /opt/spark
echo "export SPARK_HOME=/opt/spark" >> ~/.bashrc
echo "export PATH=$SPARK_HOME/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

### Step 2: Training the Model

#### Upload Training and Validation Datasets
Upload `TrainingDataset.csv` and `ValidationDataset.csv` to one of the EC2 instances.

#### Run Parallel Training
Use Spark to train the model in parallel across the 4 EC2 instances. Ensure the instances are configured to communicate with each other.

Sample command to run Spark in parallel:
```sh
/opt/spark/bin/spark-submit --master spark://<master-node-dns>:7077 --deploy-mode cluster --class <YourMainClass> <YourJarFile.jar> TrainingDataset.csv
```

### Step 3: Prediction Application

#### Upload Prediction Application
Upload the prediction application code to an EC2 instance.

#### Run Prediction
Use Spark to run the prediction application with the validation dataset to ensure it functions correctly.

Sample command to run prediction:
```sh
/opt/spark/bin/spark-submit --class <YourMainClass> <YourJarFile.jar> ValidationDataset.csv
```

### Step 4: Docker Container

#### Create Dockerfile
Create a Dockerfile to package the prediction application:
```Dockerfile
FROM openjdk:11
COPY . /app
WORKDIR /app
RUN ./gradlew build
CMD ["java", "-jar", "build/libs/<YourJarFile.jar>"]
```

#### Build and Push Docker Image
Build the Docker image and push it to Docker Hub:
```sh
docker build -t <your-dockerhub-username>/wine-quality-prediction .
docker push <your-dockerhub-username>/wine-quality-prediction
```

### Step 5: Run Prediction with Docker

#### Run Docker Container
Pull and run the Docker container on an EC2 instance:
```sh
docker pull <your-dockerhub-username>/wine-quality-prediction
docker run -it <your-dockerhub-username>/wine-quality-prediction ValidationDataset.csv
```

## Conclusion
This project demonstrates the development and deployment of a wine quality prediction model using Apache Spark and Docker on AWS. The model is trained in parallel on multiple EC2 instances and packaged in a Docker container for easy deployment. The final output includes the F1 score, which measures the prediction performance.








