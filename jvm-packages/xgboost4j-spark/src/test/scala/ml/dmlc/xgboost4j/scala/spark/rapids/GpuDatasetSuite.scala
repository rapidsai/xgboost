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

import ai.rapids.cudf.Cuda
import ml.dmlc.xgboost4j.java.spark.rapids.GpuColumnBatch
import ml.dmlc.xgboost4j.scala.spark.PerTest
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.execution.datasources.{FilePartition, PartitionedFile}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class GpuDatasetSuite extends FunSuite with PerTest {
  private lazy val TRAIN_CSV_PATH = getTestDataPath("/rank.train.csv")
  private lazy val TEST_CSV_PATH = getTestDataPath("/rank.test.csv")

  override def sparkSessionBuilder: SparkSession.Builder = SparkSession.builder()
    .master("local[1]")
    .appName("GpuDatasetSuite")
    .config("spark.ui.enabled", false)
    .config("spark.driver.memory", "512m")
    .config("spark.task.cpus", 1)

  test("findNumClasses") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader.schema(csvSchema).csv(TRAIN_CSV_PATH, TEST_CSV_PATH)
    val v = dataset.findNumClasses("e")
    assertResult(21) { v }
  }

  test("ColumnToRow") {
    assume(Cuda.isEnvCompatibleForTesting)
    val reader = new GpuDataReader(ss)
    val csvSchema = "a BOOLEAN, b DOUBLE, c DOUBLE, d DOUBLE, e INT"
    val dataset = reader
      .option("asFloats", "false")
      .schema(csvSchema)
      .csv(TRAIN_CSV_PATH, TEST_CSV_PATH)

    val rdd = dataset.buildRDD.mapPartitions((iter: Iterator[GpuColumnBatch]) => {
      val columnBatchToRow = new ColumnBatchToRow()
      while (iter.hasNext) {
        columnBatchToRow.appendColumnBatch(iter.next())
      }
      columnBatchToRow.toIterator
    })

    val data = rdd.collect()
    val firstRow = data.head
    assertResult(215) { data.length }
    assertResult(firstRow.size) { 5 }

    assertResult(false) { firstRow.getBoolean(0) }
    implicit val doubleEquality = TolerantNumerics.tolerantDoubleEquality(0.000000001)
    assert(firstRow.getDouble(1) === 985.574005058)
    assert(firstRow.getDouble(2) === 320.223538037)
    assert(firstRow.getDouble(3) === 0.621236086198)
    assert(firstRow.getInt(4) == 1)

    val secondRow = data.last
    assertResult(secondRow.size) { 5 }
    assertResult(true) {
      secondRow.getBoolean(0)
    }
    assert(secondRow.getDouble(1) === 1004.1394132)
    assert(secondRow.getDouble(2) === 464.371823646)
    assert(secondRow.getDouble(3) === 0.312492288321)
    assert(secondRow.getInt(4) == 10)
  }

  test("repartition to 1") {
    val partFiles = Array(
      PartitionedFile(null, "/a", 0, 123),
      PartitionedFile(null, "/b", 0, 456),
      PartitionedFile(null, "/c", 0, 789),
      PartitionedFile(null, "/d", 0, 2468),
      PartitionedFile(null, "/e", 0, 3579),
      PartitionedFile(null, "/f", 0, 12345),
      PartitionedFile(null, "/g", 0, 67890)
    )
    val oldPartitions = Seq(
      FilePartition(0, Seq(partFiles(0))),
      FilePartition(1, partFiles.slice(1, 4)),
      FilePartition(2, partFiles.slice(4, 6)),
      FilePartition(3, Seq(partFiles(6)))
    )
    val oldset = new GpuDataset(null, null, null, false, Integer.MAX_VALUE, Some(oldPartitions))
    val newset = oldset.repartition(1)
    assertResult(1)(newset.partitions.length)
    assertResult(0)(newset.partitions(0).index)
    val newPartFiles = newset.partitions(0).files
    assertResult(7)(newPartFiles.length)
    for (f <- partFiles) {
      assert(newPartFiles.contains(f))
    }
  }

  test("repartition") {
    val partFiles = Array(
      PartitionedFile(null, "/a", 0, 1230),
      PartitionedFile(null, "/b", 0, 4560),
      PartitionedFile(null, "/c", 0, 7890),
      PartitionedFile(null, "/d", 0, 2468),
      PartitionedFile(null, "/e", 0, 3579),
      PartitionedFile(null, "/f", 0, 12345),
      PartitionedFile(null, "/g", 0, 67890)
    )
    val oldPartitions = Seq(
      FilePartition(0, Seq(partFiles(0))),
      FilePartition(1, partFiles.slice(1, 4)),
      FilePartition(2, partFiles.slice(4, 6)),
      FilePartition(3, Seq(partFiles(6)))
    )
    val oldset = new GpuDataset(null, null, null, false, Integer.MAX_VALUE, Some(oldPartitions))
    val newset = oldset.repartition(3)
    assertResult(3)(newset.partitions.length)
    for (i <- 0 until 3) {
      assertResult(i)(newset.partitions(i).index)
    }
    assert(newset.partitions(0).files.contains(partFiles(6)))
    assert(newset.partitions(1).files.contains(partFiles(5)))
    assert(newset.partitions(2).files.contains(partFiles(2)))
    assert(newset.partitions(2).files.contains(partFiles(1)))
    assert(newset.partitions(1).files.contains(partFiles(4)))
    assert(newset.partitions(2).files.contains(partFiles(3)))
    assert(newset.partitions(2).files.contains(partFiles(0)))
  }

  test("repartition when file split is disabled") {
    assume(Cuda.isEnvCompatibleForTesting)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE"
    val testDS = new GpuDataReader(ss).schema(csvSchema).csv(getTestDataPath("/test.csvsplit.csv"))
    // Should throw exception when file split is disabled
    // but asking for more partitions than number of files.
    assertThrows[IllegalArgumentException] {
      testDS.repartition(2, false)
    }
    // Works well when file split is enabled by default
    val newPartitions = testDS.repartition(2).partitions
    assertResult(2) {newPartitions.length}
    assertResult(1) {newPartitions(0).files.length}
    assertResult(true) {
      newPartitions(1).files.head.start == newPartitions.head.files.head.length
    }
  }

  test("loading csv file when file split is disabled") {
    ss.conf.set("spark.sql.files.maxPartitionBytes", 25)
    assume(Cuda.isEnvCompatibleForTesting)
    val csvSchema = "a DOUBLE, b DOUBLE, c DOUBLE, d DOUBLE, e DOUBLE"
    // File split is enabled by default
    val testDS = new GpuDataReader(ss).schema(csvSchema)
      .csv(getTestDataPath("/test.csv"), getTestDataPath("/test.csvsplit.csv"))
    assertResult(4) { testDS.partitions.length }

    // File split is disabled
    ss.conf.set("spark.rapids.splitFile", false)
    val testDS2 = new GpuDataReader(ss).schema(csvSchema)
      .csv(getTestDataPath("/test.csv"), getTestDataPath("/test.csvsplit.csv"))
    assertResult(2) { testDS2.partitions.length }
    // Check the skew
    val rdd = testDS2.mapColumnarBatchPerPartition((iter: Iterator[GpuColumnBatch]) => {
      // for only one batch test
      if (iter.hasNext) {
        val b = iter.next()
        (0 until b.getNumRows.toInt).map { idx =>
          (0 until b.getNumColumns).map { pos =>
            b.getIntInColumn(idx, pos, 0)
          }
        }.toIterator
      } else {
        Iterator.empty
      }
    })
    val counts = rdd.collect
    assertResult(2) { rdd.getNumPartitions }
    assertResult(8) { counts.length }
    val expectedData = Seq(
      Seq(1, 2, 3, 4, 5), Seq(6, 7, 8, 9, 10), Seq(11, 12, 13, 14, 15), Seq(16, 17, 18, 19, 20),
      Seq(1, 2, 3, 4, 5), Seq(6, 7, 8, 9, 10), Seq(11, 12, 13, 14, 15), Seq(16, 17, 18, 19, 20))
    counts.zipWithIndex.foreach {
      case (row, idx) => assert(row sameElements expectedData(idx))
    }
  }

  private def getTestDataPath(resource: String): String = {
    require(resource.startsWith("/"), "resource must start with /")
    getClass.getResource(resource).getPath
  }
}
