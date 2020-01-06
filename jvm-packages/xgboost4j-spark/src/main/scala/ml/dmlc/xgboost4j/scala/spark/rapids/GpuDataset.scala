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

package ml.dmlc.xgboost4j.scala.spark.rapids

import java.util.Locale
import java.nio.charset.StandardCharsets

import ai.rapids.cudf.{DType, Table, TimeUnit}
import ml.dmlc.xgboost4j.java.XGBoostSparkJNI
import ml.dmlc.xgboost4j.java.spark.rapids.{GpuColumnBatch, PartitionReaderFactory}
import org.apache.commons.logging.LogFactory
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{BlockLocation, FileStatus, LocatedFileStatus, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.{SerializableWritable, TaskContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.UnsafeRow
import org.apache.spark.sql.execution.datasources.{FilePartition, HadoopFsRelation, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.Platform

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


/*
 * Note on compatibility with different versions of Spark:
 *
 * Some versions of Spark have slightly different internal classes. The list of
 * known ones are below. If any of these are used make sure we handle appropriately.
 *
 * PartitionDirectory:
 *   Apache Spark:
 *     case class PartitionDirectory(values: InternalRow, files: Seq[FileStatus])
 *   We have seen it overloaded as:
 *     case class PartitionDirectory(values: InternalRow, files: Seq[OtherFileStatus])
 *       Where OtherFileStatus is not a Hadoop FileStatus but has the same methods for
 *       getLen and getPath
 * FilePartition:
 *   Apache Spark:
 *     case class FilePartition(index: Int, files: Array[PartitionedFile])
 *   We have see this overloaded as:
 *     case class FilePartition(index: Int, files: Seq[PartitionedFile],
 *       preferredHosts: Seq[String])
 * PartitionedFile:
 *   Apache Spark:
 *     case class PartitionedFile(partitionValues: InternalRow,
 *       filePath: String, start: Long, length: Long, locations: Array[String])
 *   We have seen it overloaded as:
 *     case class PartitionedFile(partitionValues: InternalRow,
 *       filePath: String, start: Long, length: Long, locations: Seq[String])
 */

class GpuDataset(fsRelation: HadoopFsRelation,
    sourceType: String,
    sourceOptions: Map[String, String],
    castAllToFloats: Boolean,
    maxRowsPerChunk: Integer,
    specifiedPartitions: Option[Seq[FilePartition]] = None) {

  private val logger = LogFactory.getLog(classOf[GpuDataset])

  /** Returns the schema of the data. */
  def schema: StructType = {
    if (castAllToFloats) {
      GpuDataset.numericAsFloats(fsRelation.schema)
    } else {
      fsRelation.schema
    }
  }

  /**
    *  Return an [[RDD]] of column batches.
    *  NOTE: GpuColumnBatch is NOT serializable, so one has to be very careful in
    *        the types of operations performed on this RDD!
    */
  private[xgboost4j] def buildRDD: RDD[GpuColumnBatch] = {
    val hadoopConf = sparkSession.sparkContext.hadoopConfiguration
    val serializableConf = new SerializableWritable[Configuration](hadoopConf)
    val broadcastedConf = sparkSession.sparkContext.broadcast(serializableConf)

    val partitionReader = GpuDataset.getPartFileReader(fsRelation.sparkSession, broadcastedConf,
      sourceType, fsRelation.schema, sourceOptions, castAllToFloats, schema, maxRowsPerChunk)
    new GpuDatasetRDD(fsRelation.sparkSession, broadcastedConf, partitionReader,
      partitions, schema)
  }

  /**
    * Return an [[RDD]] by applying a function expecting RAPIDS cuDF column pointers
    * to each partition of this data.
    */
  private[xgboost4j] def mapColumnarBatchPerPartition[U: ClassTag](
      func: Iterator[GpuColumnBatch] => Iterator[U]): RDD[U] = {
    buildRDD.mapPartitions(GpuDataset.getBatchMapper(func))
  }

  /** Return a new GpuDataset that has exactly numPartitions partitions. */
  def repartition(numPartitions: Int): GpuDataset = repartition(numPartitions, true)

  def repartition(numPartitions: Int, splitFile: Boolean): GpuDataset = {
    if (numPartitions == partitions.length) {
      return new GpuDataset(fsRelation, sourceType, sourceOptions, castAllToFloats,
        maxRowsPerChunk, Some(partitions))
    }

    // build a list of all input files sorted from largest to smallest
    var files = partitions.flatMap(fp => GpuDataset.getFiles(fp))
    var permitNumPartitions = numPartitions
    files = if (files.length < numPartitions) {
      if (!splitFile) {
        throw new IllegalArgumentException("Requiring more partitions than the number of files" +
        " is NOT supported when file split is disabled for some cases, such as LTR!")
      }
      val totalSize = files.map(_.length).sum
      var newPartitionSize = totalSize / numPartitions
      if (newPartitionSize < 1) {
        // to handle corner case
        newPartitionSize = 1
        permitNumPartitions = totalSize.toInt
      }
      val splitPartFiles = ArrayBuffer[PartitionedFile]()
      for (file <- files) {
        // make sure each partfile is smaller than newPartitionSize
        if (file.length > newPartitionSize) {
          splitPartFiles ++= splitFileWhenRepartition(file, newPartitionSize)
        } else {
          splitPartFiles += file
        }
      }
      splitPartFiles
    } else {
      files
    }.sortBy(_.length)(Ordering[Long].reverse)


    // Seed the partition buckets with one of the largest files then iterate
    // through the rest of the files, adding each to the smallest bucket
    val buckets = files.take(permitNumPartitions).map(new GpuDataset.PartBucket(_))
    def bucketOrder(b: GpuDataset.PartBucket) = -b.getSize
    val queue = mutable.PriorityQueue(buckets: _*)(Ordering.by(bucketOrder))
    for (file <- files.drop(permitNumPartitions)) {
      val bucket = queue.dequeue()
      bucket.addFile(file)
      queue.enqueue(bucket)
    }

    val newPartitions = buckets.zipWithIndex.map{case (b, i) => b.toFilePartition(i)}
    new GpuDataset(fsRelation, sourceType, sourceOptions, castAllToFloats,
      maxRowsPerChunk, Some(newPartitions))
  }

  /**
    * Find the number of classes for the specified label column. The largest value within
    * the column is assumed to be one less than the number of classes.
    */
  def findNumClasses(labelCol: String): Int = {
    val fieldIndex = schema.fieldIndex(labelCol)
    val rdd = mapColumnarBatchPerPartition(GpuDataset.maxDoubleMapper(fieldIndex))
    val maxVal = rdd.reduce(Math.max)
    val numClasses = maxVal + 1
    require(numClasses.isValidInt, s"Found max label value =" +
        s" $maxVal but requires integers in range [0, ... ${Int.MaxValue})")
    numClasses.toInt
  }

  /** The SparkSession that created this GpuDataset. */
  def sparkSession: SparkSession = fsRelation.sparkSession

  private def getTotalBytes(
      partitions: Seq[PartitionDirectory],
      openCostInBytes: Long): Long = {
    partitions.flatMap { partitionDir =>
      getFileLengthsFromPartitionDir(partitionDir, openCostInBytes)
    }.sum
  }

  private def getFileLengthsFromPartitionDir(
      partitionDir: PartitionDirectory,
      openCostInBytes: Long): Seq[Long] = {
    if (partitionDir.files.nonEmpty) {
      // Should be a Hadoop FileStatus, but we have seen people overload this
      // Luckily the overloaded version has the same getLen method. Here we
      // only do the reflection when we really need to.
      if (partitionDir.files(0).isInstanceOf[FileStatus]) {
        partitionDir.files.map(_.getLen + openCostInBytes)
      } else {
        logger.debug(s"Not a FileStatus, class is: ${partitionDir.files(0).getClass}")
        partitionDir.files.asInstanceOf[Seq[Any]].map { file =>
          getLenReflection(file) + openCostInBytes
        }.asInstanceOf[Seq[Long]]
      }
    } else {
      Seq.empty[Long]
    }
  }

  private def getLenReflection(file: Any): Long = {
    val len = try {
      val getLen = file.getClass.getMethod("getLen")
      getLen.invoke(file).asInstanceOf[Long]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported File Status type ${file.getClass}, failed calling getLen"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    len.asInstanceOf[Long]
  }

  private def getPathReflection(file: Any): Path = {
    val fpath = try {
      val getPath = file.getClass.getMethod("getPath")
      getPath.invoke(file).asInstanceOf[Path]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported File Status type ${file.getClass}, failed calling getPath"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    fpath.asInstanceOf[Path]
  }


  private def fileTypeSupportSplit(fileType: String): Boolean = {
    fileType match {
      case "csv" => true
      case "parquet" => true
      case "orc" => true
      case _ => false
    }
  }

  private def getSplits(
      partitions: Seq[PartitionDirectory],
      maxSplitBytes: Long): Seq[PartitionedFile] = {
    val confSplitFile = fsRelation.sparkSession.conf
      .get("spark.rapids.splitFile", "true").toBoolean
    logger.info(s"Config 'spark.rapids.splitFile' is set to $confSplitFile")
    partitions.flatMap { partitionDir =>
      if (partitionDir.files.nonEmpty) {
        // Should be a Hadoop FileStatus, but we have seen people overload this
        // Luckily the overloaded version has the same methods. Here we
        // only do the reflection when we really need to.
        if (partitionDir.files(0).isInstanceOf[FileStatus]) {
          partitionDir.files.filter(_.getLen > 0).flatMap { file =>
            // getPath() is very expensive so we only want to call it once in this block:
            val filePath = file.getPath
            val isSplitable = fsRelation.fileFormat.isSplitable(
              fsRelation.sparkSession, fsRelation.options, filePath) &&
              fileTypeSupportSplit(sourceType) && confSplitFile
            splitFile(
              file = file,
              filePath = filePath,
              isSplitable = isSplitable,
              maxSplitBytes = maxSplitBytes,
              partitionValues = partitionDir.values
            )
          }
        } else {
          logger.debug(s"Not a FileStatus, class is: ${partitionDir.files(0).getClass}")
          val pfs = partitionDir.files.asInstanceOf[Seq[Any]].flatMap { file =>
            // skip the splitable stuff for now since not used anyway
            splitFileReflection(
              filePath = getPathReflection(file),
              fileLength = getLenReflection(file),
              partitionValues = partitionDir.values
            )
          }.asInstanceOf[Seq[PartitionedFile]]
          pfs.filter(_.length > 0)
        }
      } else {
        Seq.empty[PartitionedFile]
      }
    }.toArray.sortBy(_.length)(implicitly[Ordering[Long]].reverse)
  }

  private[rapids] lazy val partitions: Seq[FilePartition] = specifiedPartitions.getOrElse{
    val selectedPartitions = fsRelation.location.listFiles(Nil, Nil)
    val openCostInBytes = fsRelation.sparkSession.sessionState.conf.filesOpenCostInBytes
    val totalBytes = getTotalBytes(selectedPartitions, openCostInBytes)
    val maxSplitBytes = computeMaxSplitBytes(fsRelation.sparkSession, totalBytes)
    logger.info(s"Planning scan with bin packing, max size: $maxSplitBytes bytes, " +
      s"open cost is considered as scanning $openCostInBytes bytes, " +
      s"total bytes is $totalBytes.")

    val splits = getSplits(selectedPartitions, maxSplitBytes)
    val partitions = getFilePartitions(fsRelation.sparkSession, splits, maxSplitBytes)
    partitions
  }

  private def getFilePartitions(
      sparkSession: SparkSession,
      partitionedFiles: Seq[PartitionedFile],
      maxSplitBytes: Long): Seq[FilePartition] = {
    val partitions = new ArrayBuffer[FilePartition]
    val currentFiles = new ArrayBuffer[PartitionedFile]
    var currentSize = 0L

    /** Close the current partition and move to the next. */
    def closePartition(): Unit = {
      if (currentFiles.nonEmpty) {
        // Copy to a new Array.
        val newPartition =
          GpuDataset.createFilePartition(partitions.size, currentFiles.toArray.toSeq)
        partitions += newPartition
      }
      currentFiles.clear()
      currentSize = 0
    }

    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    // Assign files to partitions using "Next Fit Decreasing"
    partitionedFiles.foreach { file =>
      if (currentSize + file.length > maxSplitBytes) {
        closePartition()
      }
      // Add the given file to the current partition.
      currentSize += file.length + openCostInBytes
      currentFiles += file
    }
    closePartition()
    partitions
  }

  private def computeMaxSplitBytes(sparkSession: SparkSession,
      totalBytes: Long): Long = {
    val defaultMaxSplitBytes = sparkSession.sessionState.conf.filesMaxPartitionBytes
    val openCostInBytes = sparkSession.sessionState.conf.filesOpenCostInBytes
    val defaultParallelism = sparkSession.sparkContext.defaultParallelism
    val bytesPerCore = totalBytes / defaultParallelism

    Math.min(defaultMaxSplitBytes, Math.max(openCostInBytes, bytesPerCore))
  }

  private def splitFileReflection(
      filePath: Path,
      fileLength: Long,
      partitionValues: InternalRow): Seq[PartitionedFile] = {
    // Apache Spark open source:
    // case class PartitionedFile(partitionValues: InternalRow,
    //   filePath: String, start: Long, length: Long, locations: Array[String])
    // We have seen it overloaded as:
    // case class PartitionedFile(partitionValues: InternalRow,
    //   filePath: String, start: Long, length: Long, locations: Seq[String])
    val partitionedFile = try {
      val partedFileApply = org.apache.spark.sql.execution.datasources.PartitionedFile.getClass
        .getMethod("apply", classOf[InternalRow], classOf[String], classOf[Long],
          classOf[Long], classOf[Seq[String]])

      // not getting the file locations for now
      partedFileApply.invoke(org.apache.spark.sql.execution.datasources.PartitionedFile,
        partitionValues, filePath.toUri.toString, new java.lang.Long(0),
          new java.lang.Long(fileLength), Seq.empty[String]).asInstanceOf[PartitionedFile]
    } catch {
      case e: Exception =>
        val errorMsg = s"Unsupported PartitionedFile type, failed to create"
        throw new UnsupportedOperationException(errorMsg, e)
    }
    Seq(partitionedFile)
  }

  private def splitFile(
      file: FileStatus,
      filePath: Path,
      isSplitable: Boolean,
      maxSplitBytes: Long,
      partitionValues: InternalRow): Seq[PartitionedFile] = {

    if (isSplitable) {
      (0L until file.getLen by maxSplitBytes).map { offset =>
        val remaining = file.getLen - offset
        val size = if (remaining > maxSplitBytes) maxSplitBytes else remaining
        val hosts = getBlockHosts(getBlockLocations(file), offset, size)
        try{
          PartitionedFile(partitionValues, filePath.toUri.toString, offset, size, hosts)
        } catch {
          case e: NoSuchMethodError =>
            logger.debug("PartitionedFile, normal Apache Spark version failed.")
            // assume this is the EMR 5.28 version
            // scalastyle:off classforname
            val clzFileData = Class.forName("org.apache.spark.sql.execution.datasources.FileData")
            // scalastyle:on classforname
            val pfClass = org.apache.spark.sql.execution.datasources.PartitionedFile.getClass.
              getMethod("apply", classOf[org.apache.spark.sql.catalyst.InternalRow],
                classOf[String], classOf[Long], classOf[Long], classOf[Array[String]],
                clzFileData)
            pfClass.invoke(org.apache.spark.sql.execution.datasources.PartitionedFile,
              partitionValues, filePath.toUri.toString, java.lang.Long.valueOf(offset),
              java.lang.Long.valueOf(size), hosts, null).
              asInstanceOf[PartitionedFile]
        }
      }
    } else {
      Seq(getPartitionedFile(file, filePath, partitionValues))
    }
  }

  private def splitFileWhenRepartition(partFile: PartitionedFile,
                                        splitBytes: Long
                                      ): Seq[PartitionedFile] = {
    val partFileEnd = partFile.start + partFile.length
    (partFile.start until partFileEnd by splitBytes).map { offset =>
      val size = Math.min(partFileEnd - offset, splitBytes)
      PartitionedFile(partFile.partitionValues, partFile.filePath, offset, size, partFile.locations)
    }
  }

  private def getPartitionedFile(
      file: FileStatus,
      filePath: Path,
      partitionValues: InternalRow): PartitionedFile = {
    val hosts = getBlockHosts(getBlockLocations(file), 0, file.getLen)
    PartitionedFile(partitionValues, filePath.toUri.toString, 0, file.getLen, hosts)
  }

  private def getBlockLocations(file: FileStatus): Array[BlockLocation] = file match {
    case f: LocatedFileStatus => f.getBlockLocations
    case _ => Array.empty[BlockLocation]
  }

  // Given locations of all blocks of a single file, `blockLocations`, and an `(offset, length)`
  // pair that represents a segment of the same file, find out the block that contains the largest
  // fraction the segment, and returns location hosts of that block. If no such block can be found,
  // returns an empty array.
  private def getBlockHosts(
    blockLocations: Array[BlockLocation],
    offset: Long,
    length: Long): Array[String] = {
    val candidates = blockLocations.map {
      // The fragment starts from a position within this block
      case b if b.getOffset <= offset && offset < b.getOffset + b.getLength =>
        b.getHosts -> (b.getOffset + b.getLength - offset).min(length)

      // The fragment ends at a position within this block
      case b if offset <= b.getOffset && offset + length < b.getLength =>
        b.getHosts -> (offset + length - b.getOffset).min(length)

      // The fragment fully contains this block
      case b if offset <= b.getOffset && b.getOffset + b.getLength <= offset + length =>
        b.getHosts -> b.getLength

      // The fragment doesn't intersect with this block
      case b =>
        b.getHosts -> 0L
    }.filter { case (hosts, size) =>
      size > 0L
    }

    if (candidates.isEmpty) {
      Array.empty[String]
    } else {
      val (hosts, _) = candidates.maxBy { case (_, size) => size }
      hosts
    }
  }
}

object GpuDataset {
  private val logger = LogFactory.getLog(classOf[GpuDataset])

  // calculate bench mark
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    (result, (t1 - t0).toFloat / 1000)
  }

  private def getBatchMapper[U: ClassTag](
      func: Iterator[GpuColumnBatch] => Iterator[U]): Iterator[GpuColumnBatch] => Iterator[U] = {
    batchIter: Iterator[GpuColumnBatch] => {
      func(batchIter)
    }
  }

  private[xgboost4j] def getFiles(fp: FilePartition): Seq[PartitionedFile] = {
    fp.getClass.getMethod("files").invoke(fp) match {
      case files: Array[PartitionedFile] => Seq(files: _*)
      case files => files.asInstanceOf[Seq[PartitionedFile]]
    }
  }

  private def maxDoubleMapper(columnIndex: Int): Iterator[GpuColumnBatch] => Iterator[Double] =
    (iter: Iterator[GpuColumnBatch]) => {
      // call allocateGpuDevice to force assignment of GPU when in exclusive process mode
      // and pass that as the gpu_id, assumption is that if you are using CUDA_VISIBLE_DEVICES
      // it doesn't hurt to call allocateGpuDevice so just always do it.
      var gpuId = XGBoostSparkJNI.allocateGpuDevice()
      logger.debug("XGboost maxDoubleMapper get device: " + gpuId)

      var max: Double = Double.MinValue
      while (iter.hasNext) {
        val b = iter.next()
        val column = b.getColumnVector(columnIndex)
        val scalar = column.max()
        if (scalar.isValid) {
          val tmp = scalar.getDouble
          max = if (max < tmp) {
            tmp
          } else {
            max
          }
        }
      }
      if (max != Double.MinValue) {
        Iterator.single(max)
      } else {
        Iterator.empty
      }
    }

  def numericAsFloats(schema: StructType): StructType = {
    StructType(schema.fields.map {
      case StructField(name, nt: NumericType, nullable, metadata) =>
        StructField(name, FloatType, nullable, metadata)
      case other => other
    })
  }

  private def getPartFileReader(
        sparkSession: SparkSession,
        broadcastedConf: Broadcast[SerializableWritable[Configuration]],
        sourceType: String,
        schema: StructType,
        options: Map[String, String],
        castAllToFloats: Boolean,
        castSchema: StructType,
        maxRowsPerChunk: Integer): PartitionReaderFactory = {
    val dumpPrefix = sparkSession.sessionState.conf.
      getConfString("spark.rapids.splits.debug-dump-prefix", null)
    sourceType match {
      case "csv" =>
        GpuCSVPartitionReaderFactory(sparkSession.sessionState.conf, broadcastedConf,
          schema, schema, new StructType(), options, castAllToFloats, maxRowsPerChunk)
      case "parquet" =>
        GpuParquetPartitionReaderFactory(sparkSession.sessionState.conf, broadcastedConf, schema,
          schema, new StructType(), Array.empty, castAllToFloats,
          castSchema, dumpPrefix, maxRowsPerChunk)
      case "orc" =>
        GpuOrcPartitionReaderFactory(sparkSession.sessionState.conf, broadcastedConf, schema,
          schema, new StructType(), Array.empty, castAllToFloats, castSchema, dumpPrefix,
          maxRowsPerChunk)
      case _ => throw new UnsupportedOperationException(
        s"Unsupported source type: $sourceType")
    }
  }

  /**
   * Helper method that converts string representation of a character to actual character.
   * It handles some Java escaped strings and throws exception if given string is longer than one
   * character.
   */
  @throws[IllegalArgumentException]
  def toChar(str: String): Char = {
    (str: Seq[Char]) match {
      case Seq() => throw new IllegalArgumentException("Delimiter cannot be empty string")
      case Seq('\\') => throw new IllegalArgumentException("Single backslash is prohibited." +
        " It has special meaning as beginning of an escape sequence." +
        " To get the backslash character, pass a string with two backslashes as the delimiter.")
      case Seq(c) => c
      case Seq('\\', 't') => '\t'
      case Seq('\\', 'r') => '\r'
      case Seq('\\', 'b') => '\b'
      case Seq('\\', 'f') => '\f'
      // In case user changes quote char and uses \" as delimiter in options
      case Seq('\\', '\"') => '\"'
      case Seq('\\', '\'') => '\''
      case Seq('\\', '\\') => '\\'
      case _ if str == """\u0000""" => '\u0000'
      case Seq('\\', _) =>
        throw new IllegalArgumentException(s"Unsupported special character for delimiter: $str")
      case _ =>
        throw new IllegalArgumentException(s"Delimiter cannot be more than one character: $str")
    }
  }

  def getBool(paramName: String, paramVal: String): Boolean = {
    val lowerParamVal = paramVal.toLowerCase(Locale.ROOT)
    if (lowerParamVal == "true") {
      true
    } else if (lowerParamVal == "false") {
      false
    } else {
      throw new Exception(s"$paramName flag can be true or false")
    }
  }

  private def createFilePartition(
      size: Int,
      files: Seq[PartitionedFile]): FilePartition = {
    // Apache Spark:
    // case class FilePartition(index: Int, files: Array[PartitionedFile])
    // We have see this overloaded as:
    // case class FilePartition(index: Int, files: Seq[PartitionedFile],
    //   preferredHosts: Seq[String])
    // We skip setting the preferredHosts for overloaded version
    try {
      FilePartition(size, files)
    } catch {
      case e: NoSuchMethodError =>
        logger.debug("FilePartition, normal Apache Spark version failed")
        // assume this is the one overloaded class we know about
        try {
          val fpClass = org.apache.spark.sql.execution.datasources.FilePartition.getClass.
            getMethod("apply", classOf[Int], classOf[Seq[PartitionedFile]], classOf[Seq[String]])
          fpClass.invoke(org.apache.spark.sql.execution.datasources.FilePartition,
            new java.lang.Integer(size), files, Seq.empty[String]).asInstanceOf[FilePartition]
        } catch {
          // for spark version 2.4.4
          case _: NoSuchMethodException =>
            val fpClass = org.apache.spark.sql.execution.datasources.FilePartition.getClass.
              getMethod("apply", classOf[Int], classOf[Array[PartitionedFile]])
            fpClass.invoke(org.apache.spark.sql.execution.datasources.FilePartition,
              new java.lang.Integer(size), Array(files: _*)).asInstanceOf[FilePartition]
        }
    }
  }

  private class PartBucket(initFile: PartitionedFile) {
    private val files: ArrayBuffer[PartitionedFile] = ArrayBuffer(initFile)
    private var size = initFile.length

    def getSize: Long = size

    def addFile(file: PartitionedFile): Unit = {
      files += file
      size += file.length
    }

    def toFilePartition(index: Int): FilePartition = createFilePartition(index, files)
  }

  def getColumnRowNumberMapper: Iterator[GpuColumnBatch] => Iterator[(Int, Long)] = {
    iter: Iterator[GpuColumnBatch] => {
      var totalRows: Long = 0
      var columns: Int = 0
      var isFirstBunch = true
      while (iter.hasNext) {
        val batch = iter.next()
        totalRows += batch.getNumRows
        if (isFirstBunch) {
          isFirstBunch = false
          columns = batch.getNumColumns
        }
      }
      if (isFirstBunch) {
        Iterator.empty
      } else {
        Iterator((columns, totalRows))
      }
    }
  }

  def columnBatchToRows: Iterator[GpuColumnBatch] => Iterator[Row] = {
    iter: Iterator[GpuColumnBatch] => {
      val columnBatchToRow = new ColumnBatchToRow()
      while (iter.hasNext) {
        val batch = iter.next()
        columnBatchToRow.appendColumnBatch(batch)
      }
      columnBatchToRow.toIterator
    }
  }
}

class ColumnBatchToRow() {
  private var batches: Seq[ColumnBatchIter] = Seq()
  private lazy val batchIter = batches.toIterator
  private var currentBatchIter: ColumnBatchIter = null

  def appendColumnBatch(batch: GpuColumnBatch): Unit = {
    batches = batches :+ new ColumnBatchIter(batch)
  }

  private[xgboost4j] def toIterator: Iterator[Row] = {
    val taskContext = TaskContext.get
    val iter = new Iterator[Row] with AutoCloseable {

      override def hasNext: Boolean = {
        (currentBatchIter != null && currentBatchIter.hasNext) || nextIterator()
      }

      override def next(): Row = {
        currentBatchIter.next()
      }

      override def close(): Unit = {
        if (currentBatchIter != null) {
          currentBatchIter.close()
        }
      }

      private def nextIterator(): Boolean = {
        if (batchIter.hasNext) {
          close
          currentBatchIter = batchIter.next()
          try {
            hasNext
          }
        } else {
          false
        }
      }
    }
    taskContext.addTaskCompletionListener(_ => iter.close())
    iter
  }

  // features: indices(0) label: indices(1), label may not exist
  class ColumnBatchIter(batch: GpuColumnBatch) extends Iterator[Row] with AutoCloseable {
    val rawSchema = batch.getSchema
    val schema = StructType(rawSchema.fields.filter(x =>
      RowConverter.isSupportingType(x.dataType)))

    // schema name maps to index of schema
    val nameToIndex = schema.fieldNames.map(rawSchema.fieldIndex)

    val nativeColumnPtrs = nameToIndex.map(batch.getColumn)
    val timeUnits = nameToIndex.map(batch.getColumnVector(_).getTimeUnit)

    private val numRows = batch.getNumRows
    private val converter = new RowConverter(schema, timeUnits)
    private val rowSize = UnsafeRow.calculateBitSetWidthInBytes(schema.length) +
      schema.length * 8
    private var buffer: Long = initBuffer()
    private var nextRow = 0
    private val row = new UnsafeRow(schema.length)

    override def hasNext: Boolean = nextRow < numRows

    override def next(): Row = {
      if (nextRow >= numRows) {
        throw new NoSuchElementException
      }
      row.pointTo(null, buffer + rowSize * nextRow, rowSize)
      nextRow += 1
      converter.toExternalRow(row)
    }

    override def close(): Unit = {
      if (buffer != 0) {
        Platform.freeMemory(buffer)
        buffer = 0
      }
    }

    private def initBuffer(): Long = {
      XGBoostSparkJNI.buildUnsafeRows(nativeColumnPtrs)
    }
  }
}
