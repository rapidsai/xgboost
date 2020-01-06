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

package ml.dmlc.xgboost4j.java.spark.rapids;

import ai.rapids.cudf.*;
import org.apache.spark.sql.types.StructType;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.nio.file.Paths;
import java.util.Optional;
import java.util.ArrayList;

import static org.apache.spark.sql.types.DataTypes.FloatType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

/**
 * Test cases for GpuColumnBatchTest
 */
public class GpuColumnBatchTest {

  private final String TEST_CSV_PATH = Optional
          .ofNullable(this.getClass().getResource("/rank-weight.csv"))
          .orElseThrow(() -> new RuntimeException("Resource rank-weight.csv does not exist"))
          .getPath();
  private final String TEST_CSV_PATH2 = Optional
          .ofNullable(this.getClass().getResource("/rank-weight2.csv"))
          .orElseThrow(() -> new RuntimeException("Resource rank-weight2.csv does not exist"))
          .getPath();

  // Schemas for spark and cuDF
  private final Schema CUDF_SCHEMA = Schema.builder()
          .column(DType.FLOAT32, "label")
          .column(DType.FLOAT32, "a")
          .column(DType.FLOAT32, "b")
          .column(DType.FLOAT32, "c")
          .column(DType.INT32, "group")
          .column(DType.FLOAT32, "weight")
          .build();
  private final StructType DB_SCHEMA = new StructType()
          .add("label", FloatType)
          .add("a", FloatType)
          .add("b", FloatType)
          .add("c", FloatType)
          .add("group", IntegerType)
          .add("weight", FloatType);

  private Table mTestTable;

  @Before
  public void setup() {
    assumeTrue(Cuda.isEnvCompatibleForTesting());
    mTestTable = Table.readCSV(CUDF_SCHEMA, Paths.get(TEST_CSV_PATH).toFile());
  }

  @After
  public void tearDown() {
    if (mTestTable != null) {
      mTestTable.close();
      mTestTable = null;
    }
  }

  @Test
  public void groupAndAggregateOnColumnsHostOneTable() {
    ArrayList<Integer> groupInfo = new ArrayList();
    ArrayList<Float> weightInfo = new ArrayList();
    int groupId = 0;
    // The expected result is computed from file "rank-weight.csv" manually.
    Integer[] expectedGroupInfo = new Integer[]{7, 5, 9, 6, 6, 8, 7, 6, 5, 5};
    Float[] expectedWeightInfo = new Float[]{10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f};
    GpuColumnBatch columnBatch = new GpuColumnBatch(mTestTable, DB_SCHEMA);
    // normal case one table
    groupId = columnBatch.groupAndAggregateOnColumnsHost(4, 5, groupId,
            groupInfo, weightInfo);
    assertArrayEquals(expectedGroupInfo, groupInfo.toArray(new Integer[]{}));
    assertEquals(10, groupId);
    assertArrayEquals(expectedWeightInfo, weightInfo.toArray(new Float[]{}));

    groupInfo.clear();
    weightInfo.clear();
    groupId = 0;
    // no weight
    groupId = columnBatch.groupAndAggregateOnColumnsHost(4, -1, groupId,
            groupInfo, weightInfo);
    assertArrayEquals(expectedGroupInfo, groupInfo.toArray(new Integer[]{}));
    assertEquals(10, groupId);
    assertEquals(0, weightInfo.size());
  }

  @Test
  public void groupAndAggregateOnColumnsHostTwoTables() {
    ArrayList<Integer> groupInfo = new ArrayList();
    ArrayList<Float> weightInfo = new ArrayList();
    int groupId = 0;
    Table table2 = Table.readCSV(CUDF_SCHEMA, Paths.get(TEST_CSV_PATH2).toFile());
    // The expected result is computed from file "rank-weight.csv" and "rank-weight2.csv"
    // manually.
    Integer[] expectedGroupInfo = new Integer[]{7, 5, 9, 6, 6, 8, 7, 6, 5, 7, 7, 5, 9};
    Float[] expectedWeightInfo = new Float[]{10f, 9f, 8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f, 10f, 9f, 8f};
    GpuColumnBatch columnBatch = new GpuColumnBatch(mTestTable, DB_SCHEMA);
    GpuColumnBatch columnBatch2 = new GpuColumnBatch(table2, DB_SCHEMA);
    // normal case two tables
    groupId = columnBatch.groupAndAggregateOnColumnsHost(4, 5, groupId,
            groupInfo, weightInfo);
    groupId = columnBatch2.groupAndAggregateOnColumnsHost(4, 5, groupId,
            groupInfo, weightInfo);
    assertArrayEquals(expectedGroupInfo, groupInfo.toArray(new Integer[]{}));
    assertEquals(13, groupId);
    assertArrayEquals(expectedWeightInfo, weightInfo.toArray(new Float[]{}));

    groupInfo.clear();
    weightInfo.clear();
    groupId = 0;
    // no weight
    groupId = columnBatch.groupAndAggregateOnColumnsHost(4, -1, groupId,
            groupInfo, weightInfo);
    groupId = columnBatch2.groupAndAggregateOnColumnsHost(4, -1, groupId,
            groupInfo, weightInfo);
    assertArrayEquals(expectedGroupInfo, groupInfo.toArray(new Integer[]{}));
    assertEquals(13, groupId);
    assertEquals(0, weightInfo.size());
    table2.close();
  }
}
