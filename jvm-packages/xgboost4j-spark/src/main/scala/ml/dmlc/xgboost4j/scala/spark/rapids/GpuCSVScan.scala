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

import ai.rapids.cudf.{CSVOptions => CudfCSVOptions, HostMemoryBuffer, Table}
import ml.dmlc.xgboost4j.java.spark.rapids.{GpuColumnBatch, PartitionReader}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.compress.CompressionCodecFactory
import org.apache.spark.SerializableWritable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.execution.datasources.{HadoopFileLinesReader, PartitionedFile}
import org.apache.spark.sql.types.{StructField, StructType}
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.internal.SQLConf

case class GpuCSVPartitionReaderFactory(
    sqlConf: SQLConf,
    broadcastedConf: Broadcast[SerializableWritable[Configuration]],
    dataSchema: StructType,
    readDataSchema: StructType,
    partitionSchema: StructType, // TODO need to filter these out, or support pulling them in.
    // These are values from the file name/path itself
    options: Map[String, String],
    castAllToFloats: Boolean,
    maxReaderBatchSize: Integer) extends FilePartitionReaderFactory {

  // Try to build CSV options on the driver here to validate and fail fast.
  // The other CSV option build below will occur on the executors.
  CSVPartitionReader.buildCsvOptions(options, true) // add virtual "isFirstSplit" for validation.

  override def buildColumnarReader(partFile: PartitionedFile): PartitionReader[GpuColumnBatch] = {
    val conf = broadcastedConf.value.value

    val schema = if (castAllToFloats) {
      GpuDataset.numericAsFloats(readDataSchema)
    } else {
      readDataSchema
    }

    new CSVPartitionReader(conf, partFile, dataSchema, schema, options, maxReaderBatchSize)
  }
}

object CSVPartitionReader {
  private val csvOptionParserMap: Map[String, (CudfCSVOptions.Builder, String) => Unit] = Map(
    "comment" -> parseCSVCommentOption,
    "header" -> parseCSVHeaderOption,
    "nullValue" -> parseCSVNullValueOption,
    "quote" -> parseCSVQuoteOption,
    "sep" -> parseCSVSepOption,
    "delimiter" -> parseCSVSepOption
  )

  private def parseCSVCommentOption(b: CudfCSVOptions.Builder, v: String): Unit = {
    b.withComment(GpuDataset.toChar(v))
  }

  private def parseCSVHeaderOption(b: CudfCSVOptions.Builder, v: String): Unit = {
    if (GpuDataset.getBool("header", v)) {
      b.hasHeader
    }
  }

  private def parseCSVNullValueOption(b: CudfCSVOptions.Builder, v: String): Unit = {
    b.withNullValue(v)
  }

  private def parseCSVQuoteOption(b: CudfCSVOptions.Builder, v: String): Unit = {
    b.withQuote(GpuDataset.toChar(v))
  }

  private def parseCSVSepOption(b: CudfCSVOptions.Builder, v: String): Unit = {
    b.withDelim(GpuDataset.toChar(v))
  }

  def buildCsvOptions(options: Map[String, String], isFirstSplit: Boolean): CudfCSVOptions = {
    val builder = CudfCSVOptions.builder()
    for ((k, v) <- options) {
      if (k == "header") {
        builder.hasHeader(GpuDataset.getBool(k, v) && isFirstSplit)
      } else {
        val parseFunc = csvOptionParserMap.getOrElse(k, (_: CudfCSVOptions.Builder, _: String) => {
          throw new UnsupportedOperationException(s"CSV option $k not supported")
        })
        parseFunc(builder, v)
      }
    }
    builder.build
  }
}


class CSVPartitionReader(
    conf: Configuration,
    partFile: PartitionedFile,
    dataSchema: StructType,
    readDataSchema: StructType,
    options: Map[String, String],
    maxRowsPerChunk: Integer) extends PartitionReader[GpuColumnBatch] {
  private var batch: Option[Table] = None
  private val separator = Array('\n'.toByte)
  private val lineReader = new HadoopFileLinesReader(partFile, conf)
  private var isFirstChunkForIterator: Boolean = true
  private var isExhausted: Boolean = false

  private lazy val estimatedHostBufferSize: Long = {
    val rawPath = new Path(partFile.filePath)
    val fs = rawPath.getFileSystem(conf)
    val path = fs.makeQualified(rawPath)
    val fileSize = fs.getFileStatus(path).getLen
    val codecFactory = new CompressionCodecFactory(conf)
    val codec = codecFactory.getCodec(path)
    if (codec != null) {
      // wild guess that compression is 2X or less
      partFile.length * 2
    } else if (partFile.start + partFile.length == fileSize) {
      // last split doesn't need to read an additional record
      partFile.length
    } else {
      // wild guess for extra space needed for the record after the split end offset
      partFile.length + 128 * 1024
    }
  }

  /**
   * Grows a host buffer, returning a new buffer and closing the original
   * after copying the data into the new buffer.
   * @param original the original host memory buffer
   */
  private def growHostBuffer(original: HostMemoryBuffer, needed: Long): HostMemoryBuffer = {
    val newSize = Math.max(original.getLength * 2, needed)
    val result = HostMemoryBuffer.allocate(newSize)
    try {
      result.copyFromHostBuffer(0, original, 0, original.getLength)
      original.close()
    } catch {
      case e: Throwable =>
        result.close()
        throw e
    }
    result
  }

  private def readPartFile(): (HostMemoryBuffer, Long) = {
    isFirstChunkForIterator = false
    var succeeded = false
    var totalSize: Long = 0L
    var totalRows: Integer = 0
    var hmb = HostMemoryBuffer.allocate(estimatedHostBufferSize)
    try {
      while (lineReader.hasNext && totalRows != maxRowsPerChunk) {
        val line = lineReader.next()
        val lineSize = line.getLength
        val newTotal = totalSize + lineSize + separator.length
        if (newTotal > hmb.getLength) {
          hmb = growHostBuffer(hmb, newTotal)
        }
        // Can have an empty line, do not write this to buffer but add the separator and totalRows
        if (lineSize != 0) {
          hmb.setBytes(totalSize, line.getBytes, 0, lineSize)
        }
        hmb.setBytes(totalSize + lineSize, separator, 0, separator.length)
        totalRows += 1
        totalSize = newTotal
      }
      // Indicate this is the last chunk
      isExhausted = !lineReader.hasNext
      succeeded = true
    } finally {
      if (!succeeded) {
        hmb.close()
      }
    }
    (hmb, totalSize)
  }

  private def readBatch(): Option[Table] = {
    val hasHeader = partFile.start == 0 && isFirstChunkForIterator
    readToTable(hasHeader)
  }

  private def readToTable(isFirstSplit: Boolean): Option[Table] = {
    val (dataBuffer, dataSize) = readPartFile()
    try {
      if (dataSize == 0) {
        None
      } else {
        val csvSchemaBuilder = ai.rapids.cudf.Schema.builder
        readDataSchema.foreach(f => csvSchemaBuilder.column(
          GpuColumnBatch.getRapidsType(f.dataType), f.name))
        val newReadDataSchema: StructType = if (readDataSchema.isEmpty) {
          val smallestField = dataSchema.min(
            Ordering.by[StructField, Integer](_.dataType.defaultSize))
          StructType(Seq(smallestField))
        } else {
          readDataSchema
        }
        val csvOpts = CSVPartitionReader.buildCsvOptions(options, isFirstSplit)
        val table = Table.readCSV(csvSchemaBuilder.build(), csvOpts, dataBuffer, 0, dataSize)
        val numColumns = table.getNumberOfColumns
        if (newReadDataSchema.length != numColumns) {
          table.close()
          throw new QueryExecutionException(s"Expected ${newReadDataSchema.length} columns " +
            s"but only read ${table.getNumberOfColumns} from $partFile")
        }
        Some(table)
      }
    } finally {
      dataBuffer.close()
    }
  }

  override def next(): Boolean = {
    batch.foreach(_.close())
    batch = if (isExhausted) None else readBatch()
    batch.isDefined
  }

  override def get(): GpuColumnBatch = {
    val ret = batch.getOrElse(throw new NoSuchElementException)
    new GpuColumnBatch(ret, readDataSchema)
  }

  override def close(): Unit = {
    lineReader.close()
    batch.foreach(_.close())
    batch = None
    isExhausted = true
  }
}
