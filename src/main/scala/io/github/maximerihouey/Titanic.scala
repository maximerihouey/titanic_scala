package io.github.maximerihouey

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{IndexToString, VectorAssembler, StringIndexer}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.Pipeline


/**
  * Created by maxime on 26/10/16.
  */
object Titanic {

  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]) {
    val sparkSession = SparkSession.builder
      .master("local")
      .appName("titanic_scala")
      .getOrCreate()

    val (trainDFRaw, testDFRaw) = loadData("data/train.csv", "data/test.csv", sparkSession)
    println("Train size: %s".format(trainDFRaw.count()))
    println("Test size: %s".format(testDFRaw.count()))

    // Feature engineering
    val numericFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
    val categoricalFeatColNames = Seq("Pclass", "Sex", "Embarked")
    val consideredFeatures = numericFeatColNames ++ categoricalFeatColNames

    // Filling NaNs with average for numerical values
    val ageAvg = trainDFRaw.select("Age").union(testDFRaw.select("Age")).agg(avg("Age")).first().get(0)
    val fareAvg = trainDFRaw.select("Fare").union(testDFRaw.select("Fare")).agg(avg("Fare")).first().get(0)
    val fillNumNa = Map(
      "Fare" -> ageAvg,
      "Age" -> fareAvg
    )

    // Filling empty categorical values with most frequent value (mode)
    val EmbarkedMode = trainDFRaw.select("Embarked").union(testDFRaw.select("Embarked"))
                        .groupBy("Embarked").agg(count("Embarked").alias("count"))
                        .sort(col("count").desc).first().get(0).toString()
    val fillEmbarked: (String => String) = {
      case "" => EmbarkedMode
      case legit  => legit
    }
    val fillEmbarkedUDF = udf(fillEmbarked)

    // Applying transformations
    val trainDF = trainDFRaw.na.fill(fillNumNa).withColumn("Embarked", fillEmbarkedUDF(col("Embarked")))
    val testDF = testDFRaw.na.fill(fillNumNa).withColumn("Embarked", fillEmbarkedUDF(col("Embarked")))

    // Preparing pipeline with transformers
    val allCatData = trainDF.select(categoricalFeatColNames.map(c => col(c)): _*).union(testDF.select(categoricalFeatColNames.map(c => col(c)): _*))
    // allCatData.cache()

    // Indexing the categorical features
    val stringIndexers = categoricalFeatColNames.map { colName =>
      new StringIndexer()
        .setInputCol(colName)
        .setOutputCol(colName + "Indexed")
        .fit(allCatData)
    }

    // Indexing the survived categories
    val labelIndexer = new StringIndexer()
      .setInputCol("Survived")
      .setOutputCol("SurvivedIndexed")
      .fit(trainDF)

    // Labeling the predicted indexed survived categories
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Assembling the feature vector
    val predictionFeatures = numericFeatColNames ++ categoricalFeatColNames.map(_ + "Indexed");
    val assembler = new VectorAssembler()
      .setInputCols(Array(predictionFeatures: _*))
      .setOutputCol("Features")

    // Prediction model
    val randomForest = new RandomForestClassifier()
      .setLabelCol("SurvivedIndexed")
      .setFeaturesCol("Features")

    // define the order of the operations to be performed
    val pipeline = new Pipeline().setStages(
      stringIndexers.toArray ++ Array(labelIndexer, assembler, randomForest, labelConverter)
    )

    val predictions = pipeline.fit(trainDF).transform(testDF)

    predictions
      .withColumn("Survived", col("predictedLabel"))
      .select("PassengerId", "Survived")
      .coalesce(1)
      .write
      .format(csvFormat)
      .option("header", "true")
      .save("submission.csv")
  }

  def loadData(trainFile: String, testFile: String, sparkSession: SparkSession): (DataFrame, DataFrame) = {

    val nullable = true
    val schemaArray = Array(
      StructField("PassengerId", IntegerType, nullable),
      StructField("Survived", IntegerType, nullable),
      StructField("Pclass", IntegerType, nullable),
      StructField("Name", StringType, nullable),
      StructField("Sex", StringType, nullable),
      StructField("Age", FloatType, nullable),
      StructField("SibSp", IntegerType, nullable),
      StructField("Parch", IntegerType, nullable),
      StructField("Ticket", StringType, nullable),
      StructField("Fare", FloatType, nullable),
      StructField("Cabin", StringType, nullable),
      StructField("Embarked", StringType, nullable)
    )

    val trainSchema = StructType(schemaArray)
    val testSchema = StructType(schemaArray.filter(p => p.name != "Survived"))

    val trainDF = sparkSession.read
      .format(csvFormat)
      .option("header", "true")
      .schema(trainSchema)
      .load(trainFile)

    val testDF = sparkSession.read
      .format(csvFormat)
      .option("header", "true")
      .schema(testSchema)
      .load(testFile)

    (trainDF, testDF)
  }
}
