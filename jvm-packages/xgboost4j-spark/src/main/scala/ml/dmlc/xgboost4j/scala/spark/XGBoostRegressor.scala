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

import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch

import scala.collection.{AbstractIterator, Iterator, mutable}
import scala.collection.JavaConverters._
import ml.dmlc.xgboost4j.java.{Rabit, XGBoostSparkJNI}
import ml.dmlc.xgboost4j.{LabeledPoint => XGBLabeledPoint}
import ml.dmlc.xgboost4j.scala.spark.params.{DefaultXGBoostParamsReader, _}
import ml.dmlc.xgboost4j.scala.spark.rapids.{ColumnBatchToRow, GpuDataset, RowConverter}
import ml.dmlc.xgboost4j.scala.{Booster, DMatrix, XGBoost => SXGBoost}
import ml.dmlc.xgboost4j.scala.{EvalTrait, ObjectiveTrait}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.shared.HasWeightCol
import org.apache.spark.ml.util._
import org.apache.spark.ml._
import org.apache.spark.ml.param._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.json4s.DefaultFormats
import org.apache.spark.broadcast.Broadcast

private[spark] trait XGBoostRegressorParams extends GeneralParams with BoosterParams
  with LearningTaskParams with HasBaseMarginCol with HasWeightCol with HasGroupCol
  with ParamMapFuncs with HasLeafPredictionCol with HasContribPredictionCol with NonParamVariables
  with HasFeaturesCols

class XGBoostRegressor (
    override val uid: String,
    private val xgboostParams: Map[String, Any])
  extends Predictor[Vector, XGBoostRegressor, XGBoostRegressionModel]
    with XGBoostRegressorParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("xgbr"), Map[String, Any]())

  def this(uid: String) = this(uid, Map[String, Any]())

  def this(xgboostParams: Map[String, Any]) = this(
    Identifiable.randomUID("xgbr"), xgboostParams)

  XGBoostToMLlibParams(xgboostParams)

  def setWeightCol(value: String): this.type = set(weightCol, value)

  def setBaseMarginCol(value: String): this.type = set(baseMarginCol, value)

  def setGroupCol(value: String): this.type = set(groupCol, value)

  // setters for general params
  def setNumRound(value: Int): this.type = set(numRound, value)

  def setNumWorkers(value: Int): this.type = set(numWorkers, value)

  def setNthread(value: Int): this.type = set(nthread, value)

  def setUseExternalMemory(value: Boolean): this.type = set(useExternalMemory, value)

  def setSilent(value: Int): this.type = set(silent, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setTimeoutRequestWorkers(value: Long): this.type = set(timeoutRequestWorkers, value)

  def setCheckpointPath(value: String): this.type = set(checkpointPath, value)

  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def setEta(value: Double): this.type = set(eta, value)

  def setGamma(value: Double): this.type = set(gamma, value)

  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMinChildWeight(value: Double): this.type = set(minChildWeight, value)

  def setMaxDeltaStep(value: Double): this.type = set(maxDeltaStep, value)

  def setSubsample(value: Double): this.type = set(subsample, value)

  def setColsampleBytree(value: Double): this.type = set(colsampleBytree, value)

  def setColsampleBylevel(value: Double): this.type = set(colsampleBylevel, value)

  def setLambda(value: Double): this.type = set(lambda, value)

  def setAlpha(value: Double): this.type = set(alpha, value)

  def setTreeMethod(value: String): this.type = set(treeMethod, value)

  def setGrowPolicy(value: String): this.type = set(growPolicy, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setMaxLeaves(value: Int): this.type = set(maxLeaves, value)

  def setSketchEps(value: Double): this.type = set(sketchEps, value)

  def setScalePosWeight(value: Double): this.type = set(scalePosWeight, value)

  def setSampleType(value: String): this.type = set(sampleType, value)

  def setNormalizeType(value: String): this.type = set(normalizeType, value)

  def setRateDrop(value: Double): this.type = set(rateDrop, value)

  def setSkipDrop(value: Double): this.type = set(skipDrop, value)

  def setLambdaBias(value: Double): this.type = set(lambdaBias, value)

  // setters for learning params
  def setObjective(value: String): this.type = set(objective, value)

  def setObjectiveType(value: String): this.type = set(objectiveType, value)

  def setBaseScore(value: Double): this.type = set(baseScore, value)

  def setEvalMetric(value: String): this.type = set(evalMetric, value)

  def setTrainTestRatio(value: Double): this.type = set(trainTestRatio, value)

  def setNumEarlyStoppingRounds(value: Int): this.type = set(numEarlyStoppingRounds, value)

  def setMaximizeEvaluationMetrics(value: Boolean): this.type =
    set(maximizeEvaluationMetrics, value)

  def setCustomObj(value: ObjectiveTrait): this.type = set(customObj, value)

  def setCustomEval(value: EvalTrait): this.type = set(customEval, value)

  def setFeaturesCols(value: Seq[String]): this.type = set(featuresCols, value)

  // called at the start of fit/train when 'eval_metric' is not defined
  private def setupDefaultEvalMetric(): String = {
    require(isDefined(objective), "Users must set \'objective\' via xgboostParams.")
    if ($(objective).startsWith("rank")) {
      "map"
    } else {
      "rmse"
    }
  }

  override protected def train(dataset: Dataset[_]): XGBoostRegressionModel = {

    if (!isDefined(evalMetric) || $(evalMetric).isEmpty) {
      set(evalMetric, setupDefaultEvalMetric())
    }

    if (isDefined(customObj) && $(customObj) != null) {
      set(objectiveType, "regression")
    }

    val weight = if (!isDefined(weightCol) || $(weightCol).isEmpty) lit(1.0) else col($(weightCol))
    val baseMargin = if (!isDefined(baseMarginCol) || $(baseMarginCol).isEmpty) {
      lit(Float.NaN)
    } else {
      col($(baseMarginCol))
    }
    val group = if (!isDefined(groupCol) || $(groupCol).isEmpty) lit(-1) else col($(groupCol))
    val trainingSet: RDD[XGBLabeledPoint] = DataUtils.convertDataFrameToXGBLabeledPointRDDs(
      col($(labelCol)), col($(featuresCol)), weight, baseMargin, Some(group),
      dataset.asInstanceOf[DataFrame]).head
    val evalRDDMap = getEvalSets(xgboostParams).map {
      case (name, dataFrame) => (name,
        DataUtils.convertDataFrameToXGBLabeledPointRDDs(col($(labelCol)), col($(featuresCol)),
          weight, baseMargin, Some(group), dataFrame).head)
    }
    transformSchema(dataset.schema, logging = true)
    val derivedXGBParamMap = MLlib2XGBoostParams
    // All non-null param maps in XGBoostRegressor are in derivedXGBParamMap.
    val (_booster, _metrics) = XGBoost.trainDistributed(trainingSet, derivedXGBParamMap,
      hasGroup = group != lit(-1), evalRDDMap)
    val model = new XGBoostRegressionModel(uid, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary)
    model
  }

  override def copy(extra: ParamMap): XGBoostRegressor = defaultCopy(extra)

  def fit(dataset: GpuDataset): XGBoostRegressionModel = {
    if (!isDefined(evalMetric) || $(evalMetric).isEmpty) {
      set(evalMetric, setupDefaultEvalMetric())
    }
    if (isDefined(customObj) && $(customObj) != null) {
      set(objectiveType, "regression")
    }
    val weightColName = if (isDefined(weightCol)) $(weightCol) else null
    val groupColName = if (isDefined(groupCol)) $(groupCol) else null
    val colNames = XGBoost.buildGDFColumnNames($(featuresCols), $(labelCol),
      weightColName, groupColName)
    val derivedXGBParamMap = MLlib2XGBoostParams
    val (_booster, _metrics) = XGBoost.trainDistributedForGpuDataset(dataset, colNames,
      derivedXGBParamMap, getGpuEvalSets(xgboostParams))
    val model = new XGBoostRegressionModel(uid, _booster)
    val summary = XGBoostTrainingSummary(_metrics)
    model.setSummary(summary).setParent(this)
    copyValues(model)
  }
}

object XGBoostRegressor extends DefaultParamsReadable[XGBoostRegressor] {

  override def load(path: String): XGBoostRegressor = super.load(path)
}

class XGBoostRegressionModel private[ml] (
    override val uid: String,
    private[spark] val _booster: Booster)
  extends PredictionModel[Vector, XGBoostRegressionModel]
    with XGBoostRegressorParams with InferenceParams
    with MLWritable with Serializable {

  import XGBoostRegressionModel._

  private val logger = LogFactory.getLog("XGBoostRegressionModel")

  // only called in copy()
  def this(uid: String) = this(uid, null)

  /**
   * Get the native booster instance of this model.
   * This is used to call low-level APIs on native booster, such as "getFeatureScore".
   */
  def nativeBooster: Booster = _booster

  private var trainingSummary: Option[XGBoostTrainingSummary] = None

  /**
   * Returns summary (e.g. train/test objective history) of model on the
   * training set. An exception is thrown if no summary is available.
   */
  def summary: XGBoostTrainingSummary = trainingSummary.getOrElse {
    throw new IllegalStateException("No training summary available for this XGBoostModel")
  }

  private[spark] def setSummary(summary: XGBoostTrainingSummary): this.type = {
    trainingSummary = Some(summary)
    this
  }

  def setLeafPredictionCol(value: String): this.type = set(leafPredictionCol, value)

  def setContribPredictionCol(value: String): this.type = set(contribPredictionCol, value)

  def setTreeLimit(value: Int): this.type = set(treeLimit, value)

  def setMissing(value: Float): this.type = set(missing, value)

  def setInferBatchSize(value: Int): this.type = set(inferBatchSize, value)

  /**
   * Single instance prediction.
   * Note: The performance is not ideal, use it carefully!
   */
  override def predict(features: Vector): Double = {
    import DataUtils._
    val dm = new DMatrix(XGBoost.processMissingValues(Iterator(features.asXGB), $(missing)))
    _booster.predict(data = dm)(0)(0)
  }

  private def getMissingValue: Float = {
    val missing = getMissing
    if (!missing.isNaN && missing != 0.0f) {
      throw new RuntimeException(s"you can only specify missing value as 0.0 (the currently" +
        s" set value $missing) when you load data from GPU")
    }
    0.0f
  }

  private def transformInternal(dataset: GpuDataset): DataFrame = {
    val rawSchema = dataset.schema

    // dataReadSchema filters out some fields that RowConverter is not supporting
    val dataReadSchema = StructType(rawSchema.fields.filter(x =>
      RowConverter.isSupportingType(x.dataType)))

    val schema = StructType(dataReadSchema.fields ++
      Seq(StructField(name = _originalPredictionCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)))

    val bBooster = dataset.sparkSession.sparkContext.broadcast(_booster)
    val appName = dataset.sparkSession.sparkContext.appName

    val derivedXGBParamMap = MLlib2XGBoostParams
    val featuresColNames = derivedXGBParamMap.getOrElse("features_cols", Nil)
      .asInstanceOf[Seq[String]]

    val indices = Seq(featuresColNames.toArray).map(
      _.filter(rawSchema.fieldNames.contains).map(rawSchema.fieldIndex)
    )

    require(indices(0).length == featuresColNames.length,
      "Features column(s) in schema do NOT match the one(s) in parameters. " +
        s"Expect [${featuresColNames.mkString(", ")}], " +
        s"but found [${indices(0).map(rawSchema.fieldNames).mkString(", ")}]!")

    val missing = getMissingValue

    val resultRDD = dataset.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // call allocateGpuDevice to force assignment of GPU when in exclusive process mode
      // and pass that as the gpu_id, assumption is that if you are using CUDA_VISIBLE_DEVICES
      // it doesn't hurt to call allocateGpuDevice so just always do it.
      var gpuId = XGBoostSparkJNI.allocateGpuDevice()
      logger.info("XGboost regressor transform GPUDataSet using device: " + gpuId)
      if (gpuId == 0) {
        gpuId = -1;
      }

      val ((dm, columnBatchToRow), time) = GpuDataset.time("Transform: build dmatrix and row") {
        DataUtils.buildDMatrixIncrementally(gpuId, missing, indices, iter)
      }
      logger.debug("Benchmark [Transform: Build Dmatrix and Row] " + time)

      if (dm == null) {
        logger.info("No data for DMatrix")
        Iterator.empty
      } else {
        val rabitEnv = Array("DMLC_TASK_ID" -> TaskContext.getPartitionId().toString).toMap
        Rabit.init(rabitEnv.asJava)
        try {
          // since native model will not save predictor context, force to gpu predictor
          bBooster.value.setParam("predictor", "gpu_predictor")
          val Array(rawPredictionItr, predLeafItr, predContribItr) =
            producePredictionItrs(bBooster, dm)
          produceResultIterator(columnBatchToRow.toIterator, rawPredictionItr,
            predLeafItr, predContribItr)
        } finally {
          Rabit.shutdown()
          dm.delete()
        }
      }
    })

    bBooster.unpersist(blocking = false)
    dataset.sparkSession.createDataFrame(resultRDD, generateResultSchema(schema))
  }

  private def transformInternal(dataset: Dataset[_]): DataFrame = {

    val schema = StructType(dataset.schema.fields ++
      Seq(StructField(name = _originalPredictionCol, dataType =
        ArrayType(FloatType, containsNull = false), nullable = false)))

    val bBooster = dataset.sparkSession.sparkContext.broadcast(_booster)
    val appName = dataset.sparkSession.sparkContext.appName

    val resultRDD = dataset.asInstanceOf[Dataset[Row]].rdd.mapPartitions { rowIterator =>
      new AbstractIterator[Row] {
        private var batchCnt = 0

        private val batchIterImpl = rowIterator.grouped($(inferBatchSize)).flatMap { batchRow =>
          if (batchCnt == 0) {
            val rabitEnv = Array(
              "DMLC_TASK_ID" -> TaskContext.getPartitionId().toString,
              "DMLC_WORKER_STOP_PROCESS_ON_ERROR" -> "false").toMap
            Rabit.init(rabitEnv.asJava)
          }

          val features = batchRow.iterator.map(row => row.getAs[Vector]($(featuresCol)))

          import DataUtils._
          val cacheInfo = {
            if ($(useExternalMemory)) {
              s"$appName-${TaskContext.get().stageId()}-dtest_cache-" +
                s"${TaskContext.getPartitionId()}-batch-$batchCnt"
            } else {
              null
            }
          }

          val dm = new DMatrix(
            XGBoost.processMissingValues(features.map(_.asXGB), $(missing)),
            cacheInfo)
          try {
            val Array(rawPredictionItr, predLeafItr, predContribItr) =
              producePredictionItrs(bBooster, dm)
            produceResultIterator(batchRow.iterator, rawPredictionItr, predLeafItr, predContribItr)
          } finally {
            batchCnt += 1
            dm.delete()
          }
        }

        override def hasNext: Boolean = batchIterImpl.hasNext

        override def next(): Row = {
          val ret = batchIterImpl.next()
          if (!batchIterImpl.hasNext) {
            Rabit.shutdown()
          }
          ret
        }
      }
    }
    bBooster.unpersist(blocking = false)
    dataset.sparkSession.createDataFrame(resultRDD, generateResultSchema(schema))
  }

  private def produceResultIterator(
      originalRowItr: Iterator[Row],
      predictionItr: Iterator[Row],
      predLeafItr: Iterator[Row],
      predContribItr: Iterator[Row]): Iterator[Row] = {
    // the following implementation is to be improved
    if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(predictionItr).zip(predLeafItr).zip(predContribItr).
        map { case (((originals: Row, prediction: Row), leaves: Row), contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ leaves.toSeq ++ contribs.toSeq)
        }
    } else if (isDefined(leafPredictionCol) && $(leafPredictionCol).nonEmpty &&
      (!isDefined(contribPredictionCol) || $(contribPredictionCol).isEmpty)) {
      originalRowItr.zip(predictionItr).zip(predLeafItr).
        map { case ((originals: Row, prediction: Row), leaves: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ leaves.toSeq)
        }
    } else if ((!isDefined(leafPredictionCol) || $(leafPredictionCol).isEmpty) &&
      isDefined(contribPredictionCol) && $(contribPredictionCol).nonEmpty) {
      originalRowItr.zip(predictionItr).zip(predContribItr).
        map { case ((originals: Row, prediction: Row), contribs: Row) =>
          Row.fromSeq(originals.toSeq ++ prediction.toSeq ++ contribs.toSeq)
        }
    } else {
      originalRowItr.zip(predictionItr).map {
        case (originals: Row, originalPrediction: Row) =>
          Row.fromSeq(originals.toSeq ++ originalPrediction.toSeq)
      }
    }
  }

  private def generateResultSchema(fixedSchema: StructType): StructType = {
    var resultSchema = fixedSchema
    if (isDefined(leafPredictionCol)) {
      resultSchema = resultSchema.add(StructField(name = $(leafPredictionCol), dataType =
        ArrayType(FloatType, containsNull = false), nullable = false))
    }
    if (isDefined(contribPredictionCol)) {
      resultSchema = resultSchema.add(StructField(name = $(contribPredictionCol), dataType =
        ArrayType(FloatType, containsNull = false), nullable = false))
    }
    resultSchema
  }

  private def producePredictionItrs(booster: Broadcast[Booster], dm: DMatrix):
      Array[Iterator[Row]] = {
    val originalPredictionItr = {
      booster.value.predict(dm, outPutMargin = false, $(treeLimit)).map(Row(_)).iterator
    }
    val predLeafItr = {
      if (isDefined(leafPredictionCol)) {
        booster.value.predictLeaf(dm, $(treeLimit)).
          map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    val predContribItr = {
      if (isDefined(contribPredictionCol)) {
        booster.value.predictContrib(dm, $(treeLimit)).
          map(Row(_)).iterator
      } else {
        Iterator()
      }
    }
    Array(originalPredictionItr, predLeafItr, predContribItr)
  }

  def transform(dataset: GpuDataset): DataFrame = {
    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = transformInternal(dataset)
    var numColsOutput = 0

    val predictUDF = udf { (originalPrediction: mutable.WrappedArray[Float]) =>
      originalPrediction(0).toDouble
    }

    if ($(predictionCol).nonEmpty) {
      outputData = outputData
        .withColumn($(predictionCol), predictUDF(col(_originalPredictionCol)))
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF.drop(col(_originalPredictionCol))
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    // Output selected columns only.
    // This is a bit complicated since it tries to avoid repeated computation.
    var outputData = transformInternal(dataset)
    var numColsOutput = 0

    val predictUDF = udf { (originalPrediction: mutable.WrappedArray[Float]) =>
      originalPrediction(0).toDouble
    }

    if ($(predictionCol).nonEmpty) {
      outputData = outputData
        .withColumn($(predictionCol), predictUDF(col(_originalPredictionCol)))
      numColsOutput += 1
    }

    if (numColsOutput == 0) {
      this.logWarning(s"$uid: ProbabilisticClassificationModel.transform() was called as NOOP" +
        " since no output columns were set.")
    }
    outputData.toDF.drop(col(_originalPredictionCol))
  }

  override def copy(extra: ParamMap): XGBoostRegressionModel = {
    val newModel = copyValues(new XGBoostRegressionModel(uid, _booster), extra)
    newModel.setSummary(summary).setParent(parent)
  }

  override def write: MLWriter =
    new XGBoostRegressionModel.XGBoostRegressionModelWriter(this)
}

object XGBoostRegressionModel extends MLReadable[XGBoostRegressionModel] {

  private val _originalPredictionCol = "_originalPrediction"

  override def read: MLReader[XGBoostRegressionModel] = new XGBoostRegressionModelReader

  override def load(path: String): XGBoostRegressionModel = super.load(path)

  private[XGBoostRegressionModel]
  class XGBoostRegressionModelWriter(instance: XGBoostRegressionModel) extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      // Save metadata and Params
      implicit val format = DefaultFormats
      implicit val sc = super.sparkSession.sparkContext
      DefaultXGBoostParamsWriter.saveMetadata(instance, path, sc)
      // Save model data
      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostRegressionModel")
      val outputStream = internalPath.getFileSystem(sc.hadoopConfiguration).create(internalPath)
      instance._booster.saveModel(outputStream)
      outputStream.close()
    }
  }

  private class XGBoostRegressionModelReader extends MLReader[XGBoostRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[XGBoostRegressionModel].getName

    override def load(path: String): XGBoostRegressionModel = {
      implicit val sc = super.sparkSession.sparkContext

      val metadata = DefaultXGBoostParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString
      val internalPath = new Path(dataPath, "XGBoostRegressionModel")
      val dataInStream = internalPath.getFileSystem(sc.hadoopConfiguration).open(internalPath)

      val booster = SXGBoost.loadModel(dataInStream)
      val model = new XGBoostRegressionModel(metadata.uid, booster)
      DefaultXGBoostParamsReader.getAndSetParams(model, metadata)
      model
    }
  }
}
