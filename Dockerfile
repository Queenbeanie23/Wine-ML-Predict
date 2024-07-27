# Use an  OpenJDK runtime as a parent image
FROM openjdk:11

# Setting the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    wget https://downloads.apache.org/spark/spark-3.1.2/spark-3.1.2-bin-hadoop3.2.tgz && \
    tar xvf spark-3.1.2-bin-hadoop3.2.tgz && \
    mv spark-3.1.2-bin-hadoop3.2 /opt/spark && \
    rm spark-3.1.2-bin-hadoop3.2.tgz

# Set environment variables
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Compile the Java application
RUN javac -cp "/opt/spark/jars/*" prediction.java

# Run the prediction application
CMD ["java", "-cp", ".:/opt/spark/jars/*", "prediction"]
