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
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{FloatType, IntegerType, StructType}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class XGBoostClassifierGpuSuite extends FunSuite with PerTest {

  override def afterEach(): Unit = {
    super.afterEach()
    GpuDatasetData.classifierCleanUp()
    // Booster holds a pointer to native gpu memory. if Booster is not be disposed.
    // then Gpu memory will leak. From upstream. Booster's finalize (dispose) depends
    // on JVM GC. GC is not triggered freqently, which means gpu memory already leaks.
    // The fix is force GC in the end of each unit test.
    System.gc()
    System.runFinalization()
  }

  test("test XGBoost-Spark XGBoostClassifier setFeaturesCols") {
    val classifier = new XGBoostClassifier(Map("objective" -> "multi:softprob"))
      .setFeaturesCols(Seq("gdfCol1", "gdfCol2"))
    assert(classifier.getFeaturesCols.contains("gdfCol1"))
    assert(classifier.getFeaturesCols.contains("gdfCol2"))
    assert(classifier.getFeaturesCols.length == 2)
  }

  test("test XGBoost-Spark XGBoostClassifier wrong num_class checking") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.2f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_round" -> 30,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)
    val csvSchema = new StructType()
      .add("b", FloatType)
      .add("c", FloatType)
      .add("d", FloatType)
      .add("e", IntegerType)
    val trainDataAsGpuDS = new GpuDataReader(ss).schema(csvSchema).csv(getPath("norank.train.csv"))
    val classifier = new XGBoostClassifier(paramMap)
      .setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
      .setLabelCol("e")

    classifier.setNumClass(50)
    var result = false
    try {
      val model = classifier.fit(trainDataAsGpuDS)
      result = true
    } catch {
      case e: Throwable => {
        result = false
      }
    }
    // wrong num_class will be failed to check
    assert(result == false)
  }

  test("test XGBoost-Spark XGBoostClassifier the overloaded 'fit' should work with GpuDataset") {
    val paramMap = Map(
      "silent" -> 1,
      "eta" -> 0.2f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_round" -> 30,
      "num_workers" -> 1,
      "timeout_request_workers" -> 60000L)
    val csvSchema = new StructType()
      .add("b", FloatType)
      .add("c", FloatType)
      .add("d", FloatType)
      .add("e", IntegerType)
    val trainDataAsGpuDS = new GpuDataReader(ss).schema(csvSchema).csv(getPath("norank.train.csv"))
    val classifier = new XGBoostClassifier(paramMap)
      .setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
      .setLabelCol("e")

    // Per the training data, num classes is 21
    classifier.setNumClass(21)

    // Train without eval set(s)
    val model = classifier.fit(trainDataAsGpuDS)
    val ret = model.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < ret && ret < 20)

    // Save model to disk
    model.write.overwrite().save("/tmp/classifier/model")
    // Then read it from disk
    val lastModel = XGBoostClassificationModel.load("/tmp/classifier/model")
    val lastRet = lastModel.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < lastRet && lastRet < 20)

    // Train with eval set(s)
    val evalDataAsGpuDS = new GpuDataReader(ss).schema(csvSchema).csv(getPath("norank.eval.csv"))
    // 1) Set via xgboost ML API
    classifier.setEvalSets(Map("test" -> evalDataAsGpuDS))
    val model2 = classifier.fit(trainDataAsGpuDS)
    val ret2 = model2.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    // Allow big range since we don't care the accuracy
    assert(0 < ret2 && ret2 < 20)
    // 2) Set via param map
    val model3 = new XGBoostClassifier(paramMap ++ Array(
      "num_class" -> 21,
      "eval_sets" -> Map("test" -> evalDataAsGpuDS)))
      .setFeaturesCols(csvSchema.fieldNames.filter(_ != "e"))
      .setLabelCol("e")
      .fit(trainDataAsGpuDS)
    val ret3 = model3.predict(Vectors.dense(994.9573036, 317.483732878, 0.0313685555674))
    assert(0 < ret3 && ret3 < 20)
    assert(ret2 === ret3)
  }

  test("GPU Classifier XGBoost-Spark XGBoostClassifier output should match XGBoost4j") {
    val (trainFeaturesHandle, trainLabelsHandle) = GpuDatasetData.classifierTrain
    assert(trainFeaturesHandle.nonEmpty)
    assert(trainFeaturesHandle.size == 4)
    assert(trainLabelsHandle.nonEmpty)
    assert(trainLabelsHandle.size == 1)

    val trainingDM = new DMatrix(trainFeaturesHandle)
    trainingDM.setCUDFInfo("label", trainLabelsHandle)

    val (testFeaturesHandle, _) = GpuDatasetData.classifierTest
    assert(testFeaturesHandle.nonEmpty)
    assert(testFeaturesHandle.size == 4)

    val testDM = new DMatrix(testFeaturesHandle)

    val round = 100
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "predictor" -> "gpu_predictor",
      "max_bin" -> 16)

    val model1 = ScalaXGBoost.train(trainingDM, paramMap, round)
    val prediction1 = model1.predict(testDM)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val model2 = new XGBoostClassifier(paramMap ++ Array("num_round" -> round,
      "num_workers" -> 1))
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
      .fit(trainingDF)

    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)
    val prediction2 = model2.transform(testDF)
      .collect().map(row => row.getAs[DenseVector]("probability"))

    // the vector length in probability column is 2 since we have to fit to the evaluator in Spark
    assert(testRows === prediction2.size)
    for (i <- prediction1.indices) {
      assert(prediction1(i).length === prediction2(i).values.length)
      for (j <- prediction1(i).indices) {
        assert(prediction1(i)(j) === prediction2(i)(j))
      }
    }

    val prediction3 = model1.predict(testDM, outPutMargin = true)
    val prediction4 = model2.transform(testDF).
      collect().map(row => row.getAs[DenseVector]("rawPrediction"))

    // the vector length in rawPrediction column is 2 since we have to fit to the evaluator in Spark
    assert(testRows === prediction4.size)
    for (i <- prediction3.indices) {
      assert(prediction3(i).length === prediction4(i).values.length)
      for (j <- prediction3(i).indices) {
        assert(prediction3(i)(j) === prediction4(i)(j))
      }
    }

    trainingDM.delete()
    testDM.delete()
  }

  test("GPU Classifier Set params in XGBoost and MLlib way should produce same model") {
    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val round = 100
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16,
      "features_cols" -> featureCols,
      "label_col" -> "classIndex")

    // Set params in XGBoost way
    val model1 = new XGBoostClassifier(paramMap)
      .fit(trainingDF)

    // Set params in MLlib way
    val model2 = new XGBoostClassifier()
      .setEta(0.1f)
      .setMaxDepth(2)
      .setObjective("multi:softprob")
      .setNumRound(round)
      .setNumClass(3)
      .setNumWorkers(1)
      .setMaxBins(16)
      .setTreeMethod("gpu_hist")
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
      .fit(trainingDF)

    val prediction1 = model1.transform(testDF).select("prediction").collect()
    val prediction2 = model2.transform(testDF).select("prediction").collect()

    prediction1.zip(prediction2).foreach { case (Row(p1: Double), Row(p2: Double)) =>
      assert(p1 === p2)
    }
  }

  test("GPU Classifier test schema of XGBoostClassificationModel") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
      .fit(trainingDF)

    model.setRawPredictionCol("raw_prediction")
      .setProbabilityCol("probability_prediction")
      .setPredictionCol("final_prediction")
    var predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("sepal_length"))
    assert(predictionDF.columns.contains("classIndex"))
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("probability_prediction"))
    assert(predictionDF.columns.contains("final_prediction"))
    model.setRawPredictionCol("").setPredictionCol("final_prediction")
    predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("raw_prediction") === false)
    assert(predictionDF.columns.contains("final_prediction"))
    model.setRawPredictionCol("raw_prediction").setPredictionCol("")
    predictionDF = model.transform(testDF)
    assert(predictionDF.columns.contains("raw_prediction"))
    assert(predictionDF.columns.contains("final_prediction") === false)

    assert(model.summary.trainObjectiveHistory.length === 100)
    assert(model.summary.validationObjectiveHistory.isEmpty)
  }

  test("GPU Classifier test predictionLeaf") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
    val model = xgb.fit(trainingDF)
    model.setLeafPredictionCol("predictLeaf")
    val resultDF = model.transform(testDF)
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
  }

  test("GPU Classifier test predictionLeaf with empty column name") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val xgb = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
    val model = xgb.fit(trainingDF)
    model.setLeafPredictionCol("")
    val resultDF = model.transform(testDF)
    assert(!resultDF.columns.contains("predictLeaf"))
  }

  test("GPU Classifier test predictionContrib") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
    val model = xgb.fit(trainingDF)
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(testDF)
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictContrib"))
  }

  test("GPU Classifier test predictionContrib with empty column name") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val xgb = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
    val model = xgb.fit(trainingDF)
    model.setContribPredictionCol("")
    val resultDF = model.transform(testDF)
    assert(!resultDF.columns.contains("predictContrib"))
  }

  test("GPU Classifier test predictionLeaf and predictionContrib") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDataset(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols
    val (testDF, testRows) = GpuDatasetData.getClassifierTestGpuDataset(ss)

    val groundTruth = testRows
    val xgb = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
    val model = xgb.fit(trainingDF)
    model.setLeafPredictionCol("predictLeaf")
    model.setContribPredictionCol("predictContrib")
    val resultDF = model.transform(testDF)
    assert(resultDF.count == groundTruth)
    assert(resultDF.columns.contains("predictLeaf"))
    assert(resultDF.columns.contains("predictContrib"))
  }

  test("GPU Classifier test schema of XGBoostClassificationModel on parquet") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDatasetFromParquet(ss)
    val testDF = GpuDatasetData.getClassifierTrainGpuDatasetFromParquet(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols

    assert(testDF.schema.names.contains("A_NOT_USE"))
    assert(testDF.schema.names.contains("B_NOT_USE"))
    assert(testDF.schema.names.contains("C_NOT_USE"))
    assert(testDF.schema.names.contains("D_NOT_USE"))
    assert(testDF.schema.names.contains("line_index"))
    assert(testDF.schema.names.contains("redundance"))
    assert(testDF.schema.names.contains("sepal_length"))
    assert(testDF.schema.names.contains("sepal_width"))
    assert(testDF.schema.names.contains("petal_length"))
    assert(testDF.schema.names.contains("petal_width"))
    assert(testDF.schema.names.contains("classIndex"))
    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
      .fit(trainingDF)

    println(testDF.schema)
    val predictionDF = model.transform(testDF)
    assert(!predictionDF.columns.contains("A_NOT_USE"))
    assert(!predictionDF.columns.contains("B_NOT_USE"))
    assert(!predictionDF.columns.contains("C_NOT_USE"))
    assert(!predictionDF.columns.contains("D_NOT_USE"))
    assert(predictionDF.columns.contains("line_index"))
    assert(predictionDF.columns.contains("redundance"))
    assert(predictionDF.columns.contains("sepal_length"))
    assert(predictionDF.columns.contains("sepal_width"))
    assert(predictionDF.columns.contains("petal_length"))
    assert(predictionDF.columns.contains("petal_width"))
    assert(predictionDF.columns.contains("classIndex"))

    val value = predictionDF.head(1)

    assert(compareTwoFloats(value(0).getFloat(0), 1f))
    assert(compareTwoFloats(value(0).getFloat(1), 5.1f))
    assert(compareTwoFloats(value(0).getFloat(2), 3.5f))
    assert(compareTwoFloats(value(0).getFloat(3), 1.4f))
    assert(compareTwoFloats(value(0).getFloat(4), 0.2f))
    assert(compareTwoFloats(value(0).getFloat(5), 0.0f))
    assert(compareTwoFloats(value(0).getFloat(6), 100f))
  }

  test("GPU Classifier test schema of XGBoostClassificationModel on orc") {
    val paramMap = Map("eta" -> 0.1f,
      "max_depth" -> 2,
      "objective" -> "multi:softprob",
      "num_class" -> 3,
      "num_round" -> 100,
      "num_workers" -> 1,
      "tree_method" -> "gpu_hist",
      "max_bin" -> 16)

    val trainingDF = GpuDatasetData.getClassifierTrainGpuDatasetFromOrc(ss)
    val testDF = GpuDatasetData.getClassifierTrainGpuDatasetFromOrc(ss)
    val featureCols = GpuDatasetData.classifierFeatureCols

    assert(testDF.schema.names.contains("A_NOT_USE"))
    assert(testDF.schema.names.contains("B_NOT_USE"))
    assert(testDF.schema.names.contains("C_NOT_USE"))
    assert(testDF.schema.names.contains("D_NOT_USE"))
    assert(testDF.schema.names.contains("line_index"))
    assert(testDF.schema.names.contains("redundance"))
    assert(testDF.schema.names.contains("sepal_length"))
    assert(testDF.schema.names.contains("sepal_width"))
    assert(testDF.schema.names.contains("petal_length"))
    assert(testDF.schema.names.contains("petal_width"))
    assert(testDF.schema.names.contains("classIndex"))
    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCols(featureCols)
      .setLabelCol("classIndex")
      .fit(trainingDF)

    println(testDF.schema)
    val predictionDF = model.transform(testDF)
    assert(!predictionDF.columns.contains("A_NOT_USE"))
    assert(!predictionDF.columns.contains("B_NOT_USE"))
    assert(!predictionDF.columns.contains("C_NOT_USE"))
    assert(!predictionDF.columns.contains("D_NOT_USE"))
    assert(predictionDF.columns.contains("line_index"))
    assert(predictionDF.columns.contains("redundance"))
    assert(predictionDF.columns.contains("sepal_length"))
    assert(predictionDF.columns.contains("sepal_width"))
    assert(predictionDF.columns.contains("petal_length"))
    assert(predictionDF.columns.contains("petal_width"))
    assert(predictionDF.columns.contains("classIndex"))

    val value = predictionDF.head(1)

    assert(compareTwoFloats(value(0).getFloat(0), 1f))
    assert(compareTwoFloats(value(0).getFloat(1), 5.1f))
    assert(compareTwoFloats(value(0).getFloat(2), 3.5f))
    assert(compareTwoFloats(value(0).getFloat(3), 1.4f))
    assert(compareTwoFloats(value(0).getFloat(4), 0.2f))
    assert(compareTwoFloats(value(0).getFloat(5), 0.0f))
    assert(compareTwoFloats(value(0).getFloat(6), 100f))
  }
}
