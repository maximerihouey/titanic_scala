package io.github.maximerihouey

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types._

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
