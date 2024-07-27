import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class training {
	public static void main(String[] args) {
		SparkSession spark = SparkSession.builder()
				.appName("WineQualityPrediction")
                .master("local[*]")
				.getOrCreate();

		// Load training data
		Dataset<Row> trainingData = spark.read().format("csv")
				.option("header", "true")
				.option("inferSchema", "true")
				.load("s3://wine-set-data/TrainingDataset.csv");

		// Assemble features
		VectorAssembler assembler = new VectorAssembler()
				.setInputCols(new String[]{
						"fixed acidity", 
						"volatile acidity", 
						"citric acid", 
						"residual sugar", 
						"chlorides", 
						"free sulfur dioxide", 
						"total sulfur dioxide", 
						"density", 
						"pH", 
						"sulphates", 
						"alcohol"
				})
				.setOutputCol("features");

		Dataset<Row> trainingDataWithFeatures = assembler.transform(trainingData);

		// Train model
		LinearRegression lr = new LinearRegression()
				.setLabelCol("quality")
				.setFeaturesCol("features");

		LinearRegressionModel model = lr.fit(trainingDataWithFeatures);

		// Save model
		try {
			model.save("s3://wine-set-data/wine-quality-model");
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Load validation data
		Dataset<Row> validationData = spark.read().format("csv")
				.option("header", "true")
				.option("inferSchema", "true")
				.load("s3://wine-set-data/ValidationDataset.csv");

		Dataset<Row> validationDataWithFeatures = assembler.transform(validationData);

		// Evaluate model
		Dataset<Row> predictions = model.transform(validationDataWithFeatures);
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("quality")
				.setPredictionCol("prediction")
				.setMetricName("f1");

		double f1Score = evaluator.evaluate(predictions);
		System.out.println("F1 Score: " + f1Score);

		spark.stop();
	}
}
