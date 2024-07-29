FROM bitnami/spark:3.0.1

# Switch to root user to perform privileged operations
USER root

# Create the missing directory
RUN mkdir -p /var/lib/apt/lists/partial

# Install necessary packages
RUN apt-get update && apt-get install -y wget openjdk-11-jdk python3-pip sudo && \
    rm -rf /var/lib/apt/lists/*

# Download and extract Hadoop
RUN wget https://downloads.apache.org/hadoop/common/hadoop-3.4.0/hadoop-3.4.0.tar.gz -P /tmp && \
    tar -xzf /tmp/hadoop-3.4.0.tar.gz -C /opt && \
    mv /opt/hadoop-3.4.0 /opt/hadoop && \
    rm /tmp/hadoop-3.4.0.tar.gz

# Set Hadoop environment variables
ENV HADOOP_HOME=/opt/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin

# Download Hadoop AWS dependencies
RUN wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.0/hadoop-aws-3.2.0.jar -P /opt/bitnami/spark/jars/ && \
    wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.375/aws-java-sdk-bundle-1.11.375.jar -P /opt/bitnami/spark/jars/

# Install necessary Python packages
RUN pip3 install --no-cache-dir pandas pyspark


# Copy the Python script and dataset into the container
COPY training.py /app/predict.py
COPY TrainingDataset.csv /app/TrainingDataset.csv
COPY ValidationDataset.csv /app/ValidationDataset.csv
# Set the working directory
WORKDIR /app
# Set the entrypoint to run the Spark job
ENTRYPOINT ["spark-submit", "--master", "local[4]", "/app/prediction.py"]
