/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

package ml.dmlc.xgboost4j.scala.spark

import java.io.File

import ai.rapids.cudf.{DType, Table}
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.spark.rapids.{GpuDataReader, GpuDataset}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{BooleanType, ByteType, DataType, DateType, DoubleType, FloatType, IntegerType, LongType, ShortType, StringType, StructField, StructType, TimestampType}

object GpuDatasetData {

  private val classifierTrainDataPath = "/iris.data.csv"
  private val classifierTrainParquetDataPath = "/iris.data.parquet"
  private val classifierTrainOrcDataPath = "/iris.data.orc"
  private val classifierTestDataPath = "/iris.test.data.csv"
  private val classifierSchema = new StructType(Array(
    StructField("line_index", IntegerType),
    StructField("A_NOT_USE", StringType),
    StructField("sepal_length", FloatType),
    StructField("B_NOT_USE", StringType),
    StructField("sepal_width", FloatType),
    StructField("C_NOT_USE", StringType),
    StructField("petal_length", FloatType),
    StructField("petal_width", FloatType),
    StructField("classIndex", FloatType),
    StructField("D_NOT_USE", StringType),
    StructField("redundance", IntegerType)))

  lazy val classifierFeatureCols: Seq[String] =
    Seq("sepal_length", "sepal_width", "petal_length", "petal_width")

  def getClassifierTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, classifierSchema, classifierTrainDataPath)
  }

  def getClassifierTrainGpuDatasetFromOrc(spark: SparkSession): GpuDataset = {
    new GpuDataReader(spark).orc(getTestDataPath(classifierTrainOrcDataPath))
  }

  def getClassifierTrainGpuDatasetFromParquet(spark: SparkSession): GpuDataset = {
    new GpuDataReader(spark).parquet(getTestDataPath(classifierTrainParquetDataPath))
  }

  def getClassifierTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, classifierSchema, classifierTestDataPath)
  }

  private val regressionTrainOrcDataPath = "/norank.train.withstring.orc"
  private val regressionTrainParquetDataPath = "/norank.train.withstring.parquet"
  private val regressionTrainDataPath = "/norank.train.withstring.csv"
  private val regressionTestDataPath = "/norank.train.withstring.csv"
  private val regressionSchema = new StructType(Array(
    StructField("line_index", IntegerType),
    StructField("A_NOT_USE", StringType),
    StructField("b", DoubleType),
    StructField("B_NOT_USE", StringType),
    StructField("c", DoubleType),
    StructField("C_NOT_USE", StringType),
    StructField("d", DoubleType),
    StructField("D_NOT_USE", StringType),
    StructField("e", DoubleType),
    StructField("redundance", IntegerType)))

  lazy val regressionFeatureCols: Seq[String] =
    Seq("b", "c", "d")

  def getRegressionTrainGpuDatasetFromParquet(spark: SparkSession): GpuDataset = {
    new GpuDataReader(spark).parquet(getTestDataPath(regressionTrainParquetDataPath))
  }

  def getRegressionTrainGpuDatasetFromOrc(spark: SparkSession): GpuDataset = {
    new GpuDataReader(spark).orc(getTestDataPath(regressionTrainOrcDataPath))
  }

  def getRegressionTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, regressionSchema, regressionTrainDataPath)
  }

  def getRegressionTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, regressionSchema, regressionTestDataPath)
  }

  private val rankingTrainDataPath = "/rank.train.csv"
  private val rankingTestDataPath = "/rank.test.csv"
  private val rankingSchema = new StructType(Array(
    StructField("label", FloatType),
    StructField("b", FloatType),
    StructField("c", FloatType),
    StructField("d", FloatType),
    StructField("group", IntegerType)))

  def getRankingTrainGpuDataset(spark: SparkSession): GpuDataset = {
    getTrainGpuDataset(spark, rankingSchema, rankingTrainDataPath)
  }

  def getRankingTestGpuDataset(spark: SparkSession): (GpuDataset, Long) = {
    getTestGpuDataset(spark, rankingSchema, rankingTestDataPath)
  }

  private def getTrainGpuDataset(
      spark: SparkSession,
      schema: StructType,
      path: String): GpuDataset = {
    new GpuDataReader(spark).schema(schema).csv(getTestDataPath(path))
  }

  private def getTestGpuDataset(
      spark: SparkSession,
      schema: StructType,
      path: String): (GpuDataset, Long) = {
    val test = new GpuDataReader(spark).schema(schema).csv(getTestDataPath(path))
    val counts = test.mapColumnarBatchPerPartition(GpuDataset.getColumnRowNumberMapper).collect()

    (test, counts(0)_2)
  }

  private def getColumns(file: String, schema: StructType, features: Seq[String], label: String):
  (Table, Array[Long], Array[Long]) = {
    val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
    schema.foreach(f => csvSchemaBuilder.column(toDType(f.dataType), f.name))
    val table = Table.readCSV(csvSchemaBuilder.build(), new File(getTestDataPath(file)))
    val columnBatch = new GpuColumnBatch(table, schema)

    var needTableClose = true
    try {
      val featuresHandle = features.toArray.map(colName => {
          columnBatch.getColumn(schema.fieldIndex(colName))
      })
      val labelsHandle = Array(label).map(colName => {
          columnBatch.getColumn(schema.fieldIndex(colName))
      })
      needTableClose = false
      (table, featuresHandle, labelsHandle)
    } finally {
      if (needTableClose) {
        table.close()
      }
    }
  }

  private def toDType(dataType: DataType): DType = {
    dataType match {
      case BooleanType | ByteType => DType.INT8
      case ShortType => DType.INT16
      case IntegerType => DType.INT32
      case LongType => DType.INT64
      case FloatType => DType.FLOAT32
      case DoubleType => DType.FLOAT64
      case DateType => DType.DATE32
      case TimestampType => DType.TIMESTAMP
      case StringType => DType.STRING // TODO what do we want to do about STRING_CATEGORY???
      case unknownType => throw new UnsupportedOperationException(
        s"Unsupported Spark SQL type $unknownType")
    }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }

  def classifierCleanUp(): Unit = {
    if (classifierTrainTable != null) {
      classifierTrainTable.close()
      classifierTrainTable = null
    }

    if (classifierTestTable != null) {
      classifierTestTable.close()
      classifierTestTable = null
    }
  }

  def regressionCleanUp(): Unit = {
    if (regressionTestTable != null) {
      regressionTestTable.close()
      regressionTestTable = null
    }

    if (regressionTrainTable != null) {
      regressionTrainTable.close()
      regressionTrainTable = null
    }
  }

  var classifierTestTable: Table = null
  var classifierTrainTable: Table = null

  lazy val classifierTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTestDataPath,
      classifierSchema, classifierFeatureCols, "classIndex")
    classifierTestTable = table;
    (features, labels)
  }
  lazy val classifierTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(classifierTrainDataPath,
      classifierSchema, classifierFeatureCols, "classIndex")
    classifierTrainTable = table
    (features, labels)
  }

  var regressionTestTable: Table = null
  var regressionTrainTable: Table = null

  lazy val regressionTest: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTestDataPath,
      regressionSchema, regressionFeatureCols, "e")
    regressionTestTable = table
    (features, labels)
  }
  lazy val regressionTrain: (Array[Long], Array[Long]) = {
    val (table, features, labels) = getColumns(regressionTrainDataPath,
      regressionSchema, regressionFeatureCols, "e")
    regressionTrainTable = table
    (features, labels)
  }

  lazy val rankingFeatureCols: Seq[String] =
    rankingSchema.fieldNames.filter(_ != "group").filter(_ != "label")
}

