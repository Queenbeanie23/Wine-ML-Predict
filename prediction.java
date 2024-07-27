import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;

public class prediction {
	public static void main(String[] args) {
		// Initialize Spark session
		SparkSession spark = SparkSession.builder()
				.appName("WineQualityPrediction")
				.config("spark.master", "local")
				.config("spark.hadoop.fs.s3a.access.key", "xxxxxxxxxxxxxxxxxxxxxxxxxx")
				.config("spark.hadoop.fs.s3a.secret.key", "xxxxxxxxxxxxxxxxxxxxx")
				.config("spark.hadoop.fs.s3a.endpoint", "s3.amazonaws.com")
				.getOrCreate();

		// Load the saved model from S3
		LinearRegressionModel model = LinearRegressionModel.load("s3a://wine-set-data/saved-model");

		// Load the data you want to make predictions on from S3
		Dataset<Row> dataToPredict = spark.read().format("csv")
				.option("header", "true")
				.option("inferSchema", "true")
				.load("s3a://wine-set-data/TraingDataset.csv");

		// Assemble features into a feature vector
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

		Dataset<Row> dataWithFeatures = assembler.transform(dataToPredict);

		// Make predictions
		Dataset<Row> predictions = model.transform(dataWithFeatures);

		// Show predictions
		predictions.select("features", "prediction").show();

		// Evaluate the model using F1 score
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("label")
				.setPredictionCol("prediction")
				.setMetricName("f1");

		double f1Score = evaluator.evaluate(predictions);
		System.out.println("F1 Score: " + f1Score);

		// Stop Spark session
		spark.stop();
	}
}
