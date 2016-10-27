package io.github.maximerihouey

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

/**
  * Created by maxime on 26/10/16.
  */
object Titanic {

  val csvFormat = "com.databricks.spark.csv"

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val (trainDFRaw, testDFRaw) = loadData("data/train.csv", "data/test.csv", sqlContext)
    println("Train size: %s".format(trainDFRaw.count()))
    println("Test size: %s".format(testDFRaw.count()))

    // Feature engineering
    val numericFeatColNames = Seq("Age", "SibSp", "Parch", "Fare")
    val categoricalFeatColNames = Seq("Pclass", "Sex", "Embarked")

    // Filling NaNs with average for numerical values
    val ageAvg = trainDFRaw.select("Age").union(testDFRaw.select("Age")).agg(avg("Age")).first().get(0)
    println("Average age: %s".format(ageAvg))
    val fareAvg = trainDFRaw.select("Fare").union(testDFRaw.select("Fare")).agg(avg("Fare")).first().get(0)
    val fillNumNa = Map(
      "Fare" -> ageAvg,
      "Age" -> fareAvg
    )

    // Filling empty categorical values with most frequent value (mode)
    val EmbarkedMode = trainDFRaw.select("Embarked").union(testDFRaw.select("Embarked"))
                        .groupBy("Embarked")
                        .agg(col("Embarked"), count("Embarked")).collect()
    println("Embarked Mode")
    println(EmbarkedMode)
    println("Embarked Mode")

    val trainDF = trainDFRaw.na.fill(fillNumNa)
    val testDF = testDFRaw.na.fill(fillNumNa)

    val ageAvg2 = trainDF.select("Age").union(testDF.select("Age")).agg(avg("Age")).first().get(0)
    println("Average age2: %s".format(ageAvg2))
  }

  def loadData(trainFile: String, testFile: String, sqlContext: SQLContext): (DataFrame, DataFrame) = {

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

    val trainDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(trainSchema)
      .load(trainFile)

    val testDF = sqlContext.read
      .format(csvFormat)
      .option("header", "true")
      .schema(testSchema)
      .load(testFile)

    (trainDF, testDF)
  }
}
