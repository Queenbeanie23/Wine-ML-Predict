FROM openjdk:11
COPY . /app
WORKDIR /app
RUN ./gradlew build
CMD ["javac", "training.java"]
