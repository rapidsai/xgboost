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

import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import ml.dmlc.xgboost4j.scala.{DMatrix, XGBoost => ScalaXGBoost}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructType}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class XGBoostRegressorGpuSuite extends FunSuite with PerTest {

  override def afterEach(): Unit = {
    super.afterEach()
    GpuDatasetData.regressionCleanUp()
    // Booster holds a pointer to native gpu memory. if Booster is not be disposed.
    // then Gpu memory will leak. From upstream. Booster's finalize (dispose) depends
    // on JVM GC. GC is not triggered freqently, which means gpu memory already leaks.
    // The fix is force GC in the end of each unit test.
    System.gc()
    System.runFinalization()
  }

  test("GPU Regression test set multiple feature columns") {
    val regressor = new XGBoostRegressor(Map("objective" -> "reg:squarederror"))
      .setFeaturesCols(Seq("gdfCol1", "gdfCol2"))
    assert(regressor.getFeaturesCols.contains("gdfCol1"))
    assert(regressor.getFeaturesCols.contains("gdfCol2"))
    assert(regressor.getFeaturesCols.length == 2)
  }

  test("GPU Regression test the overloaded 'fit' should work with GpuDataset") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 50,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)
    val csvSchema = new StructType()
      .add("b", DoubleType)
      .add("c", DoubleType)
      .add("d", DoubleType)
      .add("e", IntegerType)
    // group now is not supported
    val trainDataAsGpuDS = new GpuDataReader(ss).schema(csvSchema).csv(getPath("norank.train.csv"))
    val regressor = new XGBoostRegressor(paramMap)
      .setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
      .setLabelCol("e")
    val model = regressor.fit(trainDataAsGpuDS)
    val ret = model.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < ret && ret < 20)
    // Save model to disk
    model.write.overwrite().save("/tmp/regression/model")
    // Then read it from disk
    val lastModel = XGBoostRegressionModel.load("/tmp/regression/model")
    val lastRet = lastModel.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < lastRet && lastRet < 20)

    // Train with eval set(s)
    val evalDataAsGpuDS = new GpuDataReader(ss).schema(csvSchema).csv(getPath("norank.eval.csv"))
    // 1) Set via xgboost ML API
    regressor.setEvalSets(Map("test" -> evalDataAsGpuDS))
    val model2 = regressor.fit(trainDataAsGpuDS)
    val ret2 = model2.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < ret2 && ret2 < 20)
    // 2) Set via param map
    val model3 = new XGBoostRegressor(paramMap ++ Array(
      "eval_sets" -> Map("test" -> evalDataAsGpuDS)))
      .setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
      .setLabelCol("e")
      .fit(trainDataAsGpuDS)
    val ret3 = model3.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    assert(0 < ret3 && ret3 < 20)
    assert(ret2 === ret3)
  }

  test("GPU Regression XGBoost-Spark XGBoostRegressor output should match XGBoost4j") {
    val (trainFeaturesHandle, trainLabelsHandle) = GpuDatasetData.regressionTrain
    assert(trainFeaturesHandle.nonEmpty)
    assert(trainFeaturesHandle.size == 3)
    assert(trainLabelsHandle.nonEmpty)
    assert(trainLabelsHandle.size == 1)

    val trainingDM = new DMatrix(trainFeaturesHandle)
    trainingDM.setCUDFInfo("label", trainLabelsHandle)

    val (testFeaturesHandle, _) = GpuDatasetData.regressionTest
    assert(testFeaturesHandle.nonEmpty)
    assert(testFeaturesHandle.size == 3)

    val testDM = new DMatrix(testFeaturesHandle)

    val round = 149
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L,
      "tree_method" -> "gpu_hist",
      "predictor" -> "gpu_predictor",
      "max_bin" -> 16
      )

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val model2 = new XGBoostRegressor(paramMap ++ Array("num_round" -> round,
      "num_workers" -> 1))
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
      .fit(trainingDF)

    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)
    val prediction2 = model2.transform(testDF).
      collect().map(row => row.getAs[Double]("prediction"))

    assert(prediction1.indices.count { i =>
      math.abs(prediction1(i)(0) - prediction2(i)) > 0.01
    } < prediction1.length * 0.1)

    // check the equality of single instance prediction
    val prediction3 = model1.predict(testDM)(0)(0)
    val prediction4 = model2.predict(
      Vectors.dense(985.574005058, 320.223538037, 0.621236086198))
    assert(math.abs(prediction3 - prediction4) <= 0.01f)

    trainingDM.delete()
    testDM.delete()
  }

  test("GPU Regression Set params in XGBoost and MLlib way should produce same model") {
    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L,
      "features_cols" -> featureCols,
      "label_col" -> "e")

    // Set params in XGBoost way
    val model1 = new XGBoostRegressor(paramMap)
      .fit(trainingDF)
    // Set params in MLlib way
    val model2 = new XGBoostRegressor()
      .setSilent(1)
      .setEta(0.1)
      .setMaxDepth(3)
      .setObjective("reg:squarederror")
      .setNumRound(149)
      .setNumWorkers(1)
      .setTimeoutRequestWorkers(60000)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
      .fit(trainingDF)

    val prediction1 = model1.transform(testDF).select("prediction").collect()
    val prediction2 = model2.transform(testDF).select("prediction").collect()

    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(math.abs(p1 - p2) <= 0.01f)
    }
  }

  test("GPU regression test predictionLeaf") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostRegressor(paramMap)
    val model = xgb
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
      .fit(trainingDF)
    model.setLeafPredictionCol("predictLeaf")
    val resultDF = model.transform(testDF)
    assert(resultDF.count === groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
  }

  test("GPU regression test predictionLeaf with empty column name") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val xgb = new XGBoostRegressor(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
    val model = xgb.fit(trainingDF)
    model.setLeafPredictionCol("")
    val resultDF = model.transform(testDF)
    assert(!resultDF.columns.contains("predictLeaf"))
  }

  test("GPU regression test predictionContrib") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostRegressor(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
    val model = xgb.fit(trainingDF)
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(testDF)
    assert(resultDF.count === groundTruth)
    assert(resultDF.columns.contains("predictContrib"))
  }

  test("GPU Regression test predictionContrib with empty column name") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val xgb = new XGBoostRegressor(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
    val model = xgb.fit(trainingDF)
    model.setContribPredictionCol("")
    val resultDF = model.transform(testDF)
    assert(!resultDF.columns.contains("predictContrib"))
  }

  test("GPU Regression test predictionLeaf and predictionContrib") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols
    val (testDF, testRows) = GpuDatasetData.getRegressionTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostRegressor(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
    val model = xgb.fit(trainingDF)
    model.setLeafPredictionCol("predictLeaf")
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(testDF)
    assert(resultDF.count === groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
    assert(resultDF.columns.contains("predictContrib"))
  }

  test("GPU Regression test ranking: use group data") {
    val paramMap = Map("eta" -> "1", "max_depth" -> "6", "silent" -> "1",
      "objective" -> "rank:pairwise", "num_workers" -> 1, "num_round" -> 5,
      "group_col" -> "group")

    val trainingDF = GpuDatasetData.getRankingTrainGpuDataset(ss)
    val (testDF, testRowCount) = GpuDatasetData.getRankingTestGpuDataset(ss)
    val model = new XGBoostRegressor(paramMap)
      .setFeaturesCols(GpuDatasetData.rankingFeatureCols)
      .fit(trainingDF)

    val prediction = model.transform(testDF).collect()
    assert(testRowCount === prediction.length)
  }

  test("GPU Classifier test schema of XGBoostRegressionModel on parquet") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDatasetFromParquet(ss)
    val testDF = GpuDatasetData.getRegressionTrainGpuDatasetFromParquet(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols

    assert(testDF.schema.names.contains("A_NOT_USE"))
    assert(testDF.schema.names.contains("B_NOT_USE"))
    assert(testDF.schema.names.contains("C_NOT_USE"))
    assert(testDF.schema.names.contains("D_NOT_USE"))
    assert(testDF.schema.names.contains("line_index"))
    assert(testDF.schema.names.contains("redundance"))
    assert(testDF.schema.names.contains("b"))
    assert(testDF.schema.names.contains("c"))
    assert(testDF.schema.names.contains("d"))
    assert(testDF.schema.names.contains("e"))
    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
      .fit(trainingDF)

    println(testDF.schema)
    val predictionDF = model.transform(testDF)
    assert(!predictionDF.columns.contains("A_NOT_USE"))
    assert(!predictionDF.columns.contains("B_NOT_USE"))
    assert(!predictionDF.columns.contains("C_NOT_USE"))
    assert(!predictionDF.columns.contains("D_NOT_USE"))
    assert(predictionDF.columns.contains("line_index"))
    assert(predictionDF.columns.contains("redundance"))
    assert(predictionDF.columns.contains("b"))
    assert(predictionDF.columns.contains("c"))
    assert(predictionDF.columns.contains("d"))
    assert(predictionDF.columns.contains("e"))

    val value = predictionDF.head(1)

    assert(compareTwoFloats(value(0).getFloat(0), 1f))
    assert(compareTwoFloats(value(0).getFloat(1), 985.574005058f))
    assert(compareTwoFloats(value(0).getFloat(2), 320.223538037f))
    assert(compareTwoFloats(value(0).getFloat(3), 0.621236086198f))
    assert(compareTwoFloats(value(0).getFloat(4), 1f))
    assert(compareTwoFloats(value(0).getFloat(5), 100f))
  }

  test("GPU Classifier test schema of XGBoostRegressionModel on orc") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.1f,
      "max_depth" -> 3,
      "objective" -> "reg:squarederror",
      "num_round" -> 149,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)

    val trainingDF = GpuDatasetData.getRegressionTrainGpuDatasetFromOrc(ss)
    val testDF = GpuDatasetData.getRegressionTrainGpuDatasetFromOrc(ss)
    val featureCols = GpuDatasetData.regressionFeatureCols

    assert(testDF.schema.names.contains("A_NOT_USE"))
    assert(testDF.schema.names.contains("B_NOT_USE"))
    assert(testDF.schema.names.contains("C_NOT_USE"))
    assert(testDF.schema.names.contains("D_NOT_USE"))
    assert(testDF.schema.names.contains("line_index"))
    assert(testDF.schema.names.contains("redundance"))
    assert(testDF.schema.names.contains("b"))
    assert(testDF.schema.names.contains("c"))
    assert(testDF.schema.names.contains("d"))
    assert(testDF.schema.names.contains("e"))
    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("e")
      .fit(trainingDF)

    println(testDF.schema)
    val predictionDF = model.transform(testDF)
    assert(!predictionDF.columns.contains("A_NOT_USE"))
    assert(!predictionDF.columns.contains("B_NOT_USE"))
    assert(!predictionDF.columns.contains("C_NOT_USE"))
    assert(!predictionDF.columns.contains("D_NOT_USE"))
    assert(predictionDF.columns.contains("line_index"))
    assert(predictionDF.columns.contains("redundance"))
    assert(predictionDF.columns.contains("b"))
    assert(predictionDF.columns.contains("c"))
    assert(predictionDF.columns.contains("d"))
    assert(predictionDF.columns.contains("e"))

    val value = predictionDF.head(1)

    assert(compareTwoFloats(value(0).getFloat(0), 1f))
    assert(compareTwoFloats(value(0).getFloat(1), 985.574005058f))
    assert(compareTwoFloats(value(0).getFloat(2), 320.223538037f))
    assert(compareTwoFloats(value(0).getFloat(3), 0.621236086198f))
    assert(compareTwoFloats(value(0).getFloat(4), 1f))
    assert(compareTwoFloats(value(0).getFloat(5), 100f))
  }
}
