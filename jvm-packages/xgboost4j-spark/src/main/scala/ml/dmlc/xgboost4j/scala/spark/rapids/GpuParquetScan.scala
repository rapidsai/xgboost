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

import java.io.OutputStream
import java.net.URI
import java.nio.charset.StandardCharsets
import java.util.Collections

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import ai.rapids.cudf.{ColumnVector, DType, HostMemoryBuffer, ParquetOptions, Table, TimeUnit}
import ml.dmlc.xgboost4j.java.spark.rapids.{GpuColumnBatch, PartitionReader}
import ml.dmlc.xgboost4j.scala.spark.rapids.ParquetPartitionReader.CopyRange
import org.apache.commons.io.output.{CountingOutputStream, NullOutputStream}
import org.apache.commons.io.IOUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, Path}
import org.apache.parquet.bytes.BytesUtils
import org.apache.parquet.format.converter.ParquetMetadataConverter
import org.apache.parquet.hadoop.ParquetFileReader
import org.apache.parquet.hadoop.metadata.{BlockMetaData, ColumnChunkMetaData, ColumnPath, FileMetaData, ParquetMetadata}
import org.apache.parquet.schema.{MessageType}
import org.apache.spark.SerializableWritable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.internal.Logging
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile}
import org.apache.spark.sql.execution.QueryExecutionException
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.sources.Filter
import org.apache.spark.sql.types.{NumericType, StructType}

case class GpuParquetPartitionReaderFactory(
    @transient sqlConf: SQLConf,
    broadcastedConf: Broadcast[SerializableWritable[Configuration]],
    dataSchema: StructType,
    readDataSchema: StructType,
    partitionSchema: StructType,
    filters: Array[Filter],
    castAllToFloats: Boolean,
    castSchema: StructType,
    debugDumpPrefix: String,
    maxReadBatchSize: Integer) extends FilePartitionReaderFactory with Logging {

  override def supportColumnarReads(partition: FilePartition): Boolean = true

  override def buildColumnarReader(partitionedFile: PartitionedFile):
      PartitionReader[GpuColumnBatch] = {
    buildBaseColumnarParquetReader(partitionedFile)
  }

  private def buildBaseColumnarParquetReader(file: PartitionedFile):
      PartitionReader[GpuColumnBatch] = {
    val conf = broadcastedConf.value.value
    val filePath = new Path(new URI(file.filePath))
    // noinspection ScalaDeprecation
    val footer = ParquetFileReader.readFooter(conf, filePath,
        ParquetMetadataConverter.range(file.start, file.start + file.length))
    val fileSchema = footer.getFileMetaData.getSchema
    val blocks = footer.getBlocks
    val columnPaths = fileSchema.getPaths.asScala.map(x => ColumnPath.get(x: _*))
    val clippedBlocks = ParquetPartitionReader.clipBlocks(columnPaths, blocks.asScala)
    // val debugDumpPrefix = conf.get("spark.rapids.splits.debug-dump-prefix", null)
    if (clippedBlocks.length == 0) {
      val end = file.start + file.length
      logWarning("Couldn't find parquet row groups from file: " + file.filePath +
        ", range: " + file.start + "-"  + end)
    }
    new ParquetPartitionReader(conf, file, filePath, clippedBlocks, fileSchema,
      dataSchema, debugDumpPrefix, castAllToFloats, castSchema, maxReadBatchSize)
  }
}

/**
  * A PartitionReader that reads a Parquet file split on the GPU.
  *
  * Efficiently reading a Parquet split on the GPU requires re-constructing the Parquet file
  * in memory that contains just the column chunks that are needed. This avoids sending
  * unnecessary data to the GPU and saves GPU memory.
  *
  * @param conf the Hadoop configuration
  * @param split the file split to read
  * @param filePath the path to the Parquet file
  * @param clippedBlocks the block metadata from the original Parquet file that has been clipped
  *                      to only contain the column chunks to be read
  * @param clippedParquetSchema the Parquet schema from the original Parquet file that has been
  *                             clipped to contain only the columns to be read
  * @param readDataSchema the Spark schema describing what will be read
  * @param debugDumpPrefix a path prefix to use for dumping the fabricated Parquet data or null
  */
class ParquetPartitionReader(
    conf: Configuration,
    split: PartitionedFile,
    filePath: Path,
    clippedBlocks: Seq[BlockMetaData],
    clippedParquetSchema: MessageType,
    readDataSchema: StructType,
    debugDumpPrefix: String,
    castToFloat: Boolean,
    castSchema: StructType,
    maxReadBatchSize: Integer) extends PartitionReader[GpuColumnBatch] with Logging {
  private var isExhausted: Boolean = false
  private var batch: Option[Table] = None
  private val blockIterator: BufferedIterator[BlockMetaData] = clippedBlocks.iterator.buffered
  private val copyBufferSize = conf.getInt("parquet.read.allocation.size", 8 * 1024 * 1024)

  override def next(): Boolean = {
    batch.foreach(_.close())
    batch = None
    if (!isExhausted) {
      if (!blockIterator.hasNext) {
        isExhausted = true
      } else {
        batch = readBatch()
      }
    }
    batch.isDefined
  }

  override def get(): GpuColumnBatch = {
    val ret = batch.getOrElse(throw new NoSuchElementException)
    var schema: StructType = readDataSchema
    if (castToFloat) {
      schema = castSchema
    }
    new GpuColumnBatch(ret, schema)
  }

  override def close(): Unit = {
    batch.foreach(_.close())
    batch = None
    isExhausted = true
  }

  private def readPartFile(blocks: Seq[BlockMetaData]): (HostMemoryBuffer, Long) = {
    val in = filePath.getFileSystem(conf).open(filePath)
    try {
      var succeeded = false
      val hmb = HostMemoryBuffer.allocate(calculateParquetOutputSize(blocks))
      try {
        val out = new HostMemoryOutputStream(hmb)
        out.write(ParquetPartitionReader.PARQUET_MAGIC)
        val outputBlocks = copyBlocksData(in, out, blocks)
        val footerPos = out.getPos
        writeFooter(out, outputBlocks)
        BytesUtils.writeIntLittleEndian(out, (out.getPos - footerPos).toInt)
        out.write(ParquetPartitionReader.PARQUET_MAGIC)
        succeeded = true
        (hmb, out.getPos)
      } finally {
        if (!succeeded) {
          hmb.close()
        }
      }
    } finally {
      in.close()
    }
  }

  private def calculateParquetOutputSize(currentChunkedBlocks: Seq[BlockMetaData]): Long = {
    // start with the size of Parquet magic (at start+end) and footer length values
    var size: Long = 4 + 4 + 4

    // add in the size of the row group data

    // Calculate the total amount of column data that will be copied
    // NOTE: Avoid using block.getTotalByteSize here as that is the
    //       uncompressed size rather than the size in the file.
    size += currentChunkedBlocks.flatMap(_.getColumns.asScala.map(_.getTotalSize)).sum

    // Calculate size of the footer metadata.
    // This uses the column metadata from the original file, but that should
    // always be at least as big as the updated metadata in the output.
    val out = new CountingOutputStream(new NullOutputStream)
    writeFooter(out, currentChunkedBlocks)
    size + out.getByteCount
  }

  private def writeFooter(out: OutputStream, blocks: Seq[BlockMetaData]): Unit = {
    val fileMeta = new FileMetaData(clippedParquetSchema, Collections.emptyMap[String, String],
      ParquetPartitionReader.PARQUET_CREATOR)
    val metadataConverter = new ParquetMetadataConverter
    val footer = new ParquetMetadata(fileMeta, blocks.asJava)
    val meta = metadataConverter.toParquetMetadata(ParquetPartitionReader.PARQUET_VERSION, footer)
    org.apache.parquet.format.Util.writeFileMetaData(meta, out)
  }

  private def copyDataRange(
      range: CopyRange,
      in: FSDataInputStream,
      out: OutputStream,
      copyBuffer: Array[Byte]): Unit = {
    if (in.getPos != range.offset) {
      in.seek(range.offset)
    }
    var bytesLeft = range.length
    while (bytesLeft > 0) {
      // downcast is safe because copyBuffer.length is an int
      val readLength = Math.min(bytesLeft, copyBuffer.length).toInt
      in.readFully(copyBuffer, 0, readLength)
      out.write(copyBuffer, 0, readLength)
      bytesLeft -= readLength
    }
  }

  /**
    * Copies the data corresponding to the clipped blocks in the original file and compute the
    * block metadata for the output. The output blocks will contain the same column chunk
    * metadata but with the file offsets updated to reflect the new position of the column data
    * as written to the output.
    *
    * @param in the input stream for the original Parquet file
    * @param out the output stream to receive the data
    * @return updated block metadata corresponding to the output
    */
  private def copyBlocksData(
      in: FSDataInputStream,
      out: HostMemoryOutputStream,
      blocks: Seq[BlockMetaData]): Seq[BlockMetaData] = {
    var totalRows: Long = 0
    val outputBlocks = new ArrayBuffer[BlockMetaData](blocks.length)
    val copyRanges = new ArrayBuffer[CopyRange]
    var currentCopyStart = 0L
    var currentCopyEnd = 0L
    var totalBytesToCopy = 0L
    blocks.foreach { block =>
      totalRows += block.getRowCount
      val columns = block.getColumns.asScala
      val outputColumns = new ArrayBuffer[ColumnChunkMetaData](columns.length)
      columns.foreach { column =>
        // update column metadata to reflect new position in the output file
        val offsetAdjustment = out.getPos + totalBytesToCopy - column.getStartingPos
        val newDictOffset = if (column.getDictionaryPageOffset > 0) {
          column.getDictionaryPageOffset + offsetAdjustment
        } else {
          0
        }
        // noinspection ScalaDeprecation
        outputColumns += ColumnChunkMetaData.get(
          column.getPath,
          column.getPrimitiveType,
          column.getCodec,
          column.getEncodingStats,
          column.getEncodings,
          column.getStatistics,
          column.getStartingPos + offsetAdjustment,
          newDictOffset,
          column.getValueCount,
          column.getTotalSize,
          column.getTotalUncompressedSize)

        if (currentCopyEnd != column.getStartingPos) {
          if (currentCopyEnd != 0) {
            copyRanges.append(CopyRange(currentCopyStart, currentCopyEnd - currentCopyStart))
          }
          currentCopyStart = column.getStartingPos
          currentCopyEnd = currentCopyStart
        }
        currentCopyEnd += column.getTotalSize
        totalBytesToCopy += column.getTotalSize
      }
      outputBlocks += ParquetPartitionReader.newParquetBlock(block.getRowCount, outputColumns)
    }

    if (currentCopyEnd != currentCopyStart) {
      copyRanges.append(CopyRange(currentCopyStart, currentCopyEnd - currentCopyStart))
    }
    val copyBuffer = new Array[Byte](copyBufferSize)
    copyRanges.foreach(copyRange => copyDataRange(copyRange, in, out, copyBuffer))
    outputBlocks
  }

  private def readBatch(): Option[Table] = {
    val currentChunkedBlocks = populateCurrentBlockChunk()
    if (readDataSchema.isEmpty) {
      // not reading any data, so return a degenerate ColumnarBatch with the row count
      val numRows = currentChunkedBlocks.map(_.getRowCount).sum.toInt
      if (numRows == 0) {
        None
      } else {
        // Some(new ColumnarBatch(Array.empty, numRows.toInt))
        None
      }
    } else {
      var success = false
      var table: Option[Table] = None

      try {
        table = readToTable(currentChunkedBlocks)
        if (castToFloat && table.isDefined) {
          val numColumns = table.get.getNumberOfColumns
          val columns = new Array[ColumnVector](table.get.getNumberOfColumns)
          try {
            for (i <- 0 until numColumns) {
              val c = table.get.getColumn(i)
              columns(i) = readDataSchema.fields(i).dataType match {
                case nt: NumericType => c.asFloats()
                case _ => c.incRefCount()
              }
            }
            val tmp = table.get
            table = Some(new Table(columns: _*))
            tmp.close()
            success = true
          } finally {
            columns.foreach(v => if (v != null) v.close())
          }
        } else {
          success = true
        }
        table
      } finally {
        if (!success) {
          table.foreach(_.close())
        }
      }
    }
  }

  private def readToTable(currentChunkedBlocks: Seq[BlockMetaData]): Option[Table] = {
    if (currentChunkedBlocks.isEmpty) {
      return None
    }

    val (dataBuffer, dataSize) = readPartFile(currentChunkedBlocks)
    try {
      if (dataSize == 0) {
        None
      } else {
        if (debugDumpPrefix != null) {
          dumpParquetData(dataBuffer, dataSize)
        }
        val parseOpts = ParquetOptions.builder().includeColumn(
            readDataSchema.fieldNames: _*).build()
        val table = Table.readParquet(parseOpts, dataBuffer, 0, dataSize)
        val numColumns = table.getNumberOfColumns
        if (readDataSchema.length != numColumns) {
          table.close()
          throw new QueryExecutionException(s"Expected ${readDataSchema.length} columns " +
              s"but read ${table.getNumberOfColumns} from $filePath")
        }
        Some(handleDate64Casts(table))
      }
    } finally {
      dataBuffer.close()
    }
  }


  // The GPU Parquet reader always casts timestamp columns to DATE64.
  // Cast DATE64 columns back to TIMESTAMP
  private def handleDate64Casts(table: Table): Table = {
    var columns: ArrayBuffer[ColumnVector] = null
    // If we have to create a new column from the cast, we need to close it after adding it to the
    // table, which will increment its reference count
    var toClose = new ArrayBuffer[ColumnVector]()
    try {
      for (i <- 0 until table.getNumberOfColumns) {
        val column = table.getColumn(i)
        if (column.getType == DType.DATE64) {
          if (columns == null) {
            columns = (0 until table.getNumberOfColumns).map(table.getColumn).to[ArrayBuffer]
          }
          val rapidsType = GpuColumnBatch.getRapidsType(readDataSchema.fields(i).dataType)
          columns(i) = columns(i).castTo(rapidsType, TimeUnit.MICROSECONDS)
          toClose += columns(i)
        }
      }

      var result = table
      if (columns != null) {
        try {
          result = new Table(columns: _*)
        } finally {
          table.close()
        }
      }

      result
    } finally {
      toClose.foreach(_.close())
    }
  }

  private def populateCurrentBlockChunk(): Seq[BlockMetaData] = {
    val currentChunk = new ArrayBuffer[BlockMetaData]
    if (blockIterator.hasNext) {
      // return at least one block even if it is larger than the configured limit
      currentChunk += blockIterator.next
      var numRows = currentChunk.head.getRowCount
      if (numRows > Integer.MAX_VALUE) {
        throw new UnsupportedOperationException("Too many rows in split")
      }

      while (blockIterator.hasNext
        && blockIterator.head.getRowCount + numRows <= maxReadBatchSize) {
        currentChunk += blockIterator.next
        numRows += currentChunk.last.getRowCount
      }
    }
    currentChunk
  }

  private def dumpParquetData(
      hmb: HostMemoryBuffer,
      dataLength: Long): Unit = {
    val (out, path) = FileUtils.createTempFile(conf, debugDumpPrefix, ".parquet")
    try {
      logInfo(s"Writing Parquet split data for $split to $path")
      val in = new HostMemoryInputStream(hmb, dataLength)
      IOUtils.copy(in, out)
    } finally {
      out.close()
    }
  }
}

object ParquetPartitionReader {
  private val PARQUET_MAGIC = "PAR1".getBytes(StandardCharsets.US_ASCII)
  private val PARQUET_CREATOR = "RAPIDS GpuDataset"
  private val PARQUET_VERSION = 1

  private case class CopyRange(offset: Long, length: Long)

  /**
    * Build a new BlockMetaData
    *
    * @param rowCount the number of rows in this block
    * @param columns the new column chunks to reference in the new BlockMetaData
    * @return the new BlockMetaData
    */
  private def newParquetBlock(
      rowCount: Long,
      columns: Seq[ColumnChunkMetaData]): BlockMetaData = {
    val block = new BlockMetaData
    block.setRowCount(rowCount)

    var totalSize: Long = 0
    columns.foreach { column =>
      block.addColumn(column)
      totalSize += column.getTotalUncompressedSize
    }
    block.setTotalByteSize(totalSize)

    block
  }

  /**
    * Trim block metadata to contain only the column chunks that occur in the specified columns.
    * The column chunks that are returned are preserved verbatim
    * (i.e.: file offsets remain unchanged).
    *
    * @param columnPaths the paths of columns to preserve
    * @param blocks the block metadata from the original Parquet file
    * @return the updated block metadata with undesired column chunks removed
    */
  private[spark] def clipBlocks(columnPaths: Seq[ColumnPath], blocks: Seq[BlockMetaData]):
      Seq[BlockMetaData] = {
    val pathSet = columnPaths.toSet
    blocks.map(oldBlock => {
      // noinspection ScalaDeprecation
      val newColumns = oldBlock.getColumns.asScala.filter(c => pathSet.contains(c.getPath))
      ParquetPartitionReader.newParquetBlock(oldBlock.getRowCount, newColumns)
    })
  }
}


