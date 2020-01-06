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

import java.util.List;

import ai.rapids.cudf.ColumnVector;

import ai.rapids.cudf.DType;
import ai.rapids.cudf.Table;

import org.apache.spark.sql.types.*;

public class GpuColumnBatch {
  private final Table table;
  private final StructType schema;

  public GpuColumnBatch(Table table, StructType schema) {
    this.table = table;
    this.schema = schema;
  }

  public StructType getSchema() {
    return schema;
  }

  public long getNumRows() {
    return table.getRowCount();
  }

  public int getNumColumns() {
    return table.getNumberOfColumns();
  }

  public ColumnVector getColumnVector(int index) {
    return table.getColumn(index);
  }

  public long getColumn(int index) {
    ColumnVector v = table.getColumn(index);
    return v.getNativeCudfColumnAddress();
  }

  public ColumnVector getColumnVectorInitHost(int index) {
    ColumnVector cv = table.getColumn(index);
    cv.ensureOnHost();
    return cv;
  }

  private double getNumericValueInColumn(int dataIndex, ColumnVector cv, StructField field) {
    DataType type = field.dataType();
    double value;
    if (type instanceof FloatType) {
      value = cv.getFloat(dataIndex);
    } else if (type instanceof IntegerType) {
      value = cv.getInt(dataIndex);
    } else if (type instanceof ByteType) {
      value = cv.getByte(dataIndex);
    } else if (type instanceof ShortType) {
      value = cv.getShort(dataIndex);
    } else if (type instanceof DoubleType) {
      value = cv.getDouble(dataIndex);
    } else if (type instanceof LongType) {
      value = cv.getLong(dataIndex);
    } else {
      throw new IllegalArgumentException("Not a numeric type in column: " + field.name());
    }
    return value;
  }

  private double getNumericValueInColumn(int dataIndex, int colIndex, double defVal) {
    ColumnVector cv = getColumnVector(colIndex);
    cv.ensureOnHost();
    return cv.getRowCount() > 0 ?
            getNumericValueInColumn(dataIndex, cv, getSchema().apply(colIndex)) :
            defVal;
  }

  public int getIntInColumn(int dataIndex, int colIndex, int defVal) {
    return (int)getNumericValueInColumn(dataIndex, colIndex, defVal);
  }

  /**
   * Group data by column "groupIndex", and do "count" aggregation on column "groupIndex",
   * while do aggregation similar to "average" on column "weightIdx", but require values in
   * a group are equal to each other, then merge the results with "groupInfo" and "weightInfo"
   * separately.
   *
   * This is used to calculate group and weight info, and support chunk loading.
   *
   * @param groupIdx The index of column to group by.
   * @param weightIdx The index of column where to get a value in each group.
   * @param prevTailGid The group id of last group in prevGroupInfo.
   * @param groupInfo Group information calculated from earlier batches.
   * @param weightInfo Weight information calculated from earlier batches.
   * @return The group id of last group in current column batch.
   */
  public int groupAndAggregateOnColumnsHost(int groupIdx, int weightIdx, int prevTailGid,
      List<Integer> groupInfo, List<Float> weightInfo) {
    // Weight: Initialize info if having weight column
    final boolean hasWeight = weightIdx >= 0;
    ColumnVector aggrCV = null;
    Float curWeight = null;
    if (hasWeight) {
      aggrCV = getColumnVectorInitHost(weightIdx);
      Float firstWeight = aggrCV.getRowCount() > 0 ?
              (float)getNumericValueInColumn(0, aggrCV, getSchema().apply(weightIdx)) : null;
      curWeight = weightInfo.isEmpty() ? firstWeight : weightInfo.get(weightInfo.size() - 1);
    }
    // Initialize group info
    ColumnVector groupCV = getColumnVectorInitHost(groupIdx);
    StructField groupSF = getSchema().apply(groupIdx);
    int groupId = prevTailGid;
    int groupSize = groupInfo.isEmpty() ? 0 : groupInfo.get(groupInfo.size() - 1);
    for (int i = 0; i < groupCV.getRowCount(); i ++) {
      Float weight = hasWeight ?
              (float)getNumericValueInColumn(i, aggrCV, getSchema().apply(weightIdx)) : 0;
      int gid = (int)getNumericValueInColumn(i, groupCV, groupSF);
      if(gid == groupId) {
        // The same group
        groupSize ++;
        // Weight: Check values in the same group if having weight column
        if (hasWeight && !weight.equals(curWeight)) {
          throw new IllegalArgumentException("The instances in the same group have to be" +
                  " assigned with the same weight. Unexpected weight: " + weight);
        }
      } else {
        // A new group, update group info
        addOrUpdateInfos(prevTailGid, groupId, groupSize, curWeight, hasWeight, groupInfo,
                weightInfo);
        if (hasWeight) {
          curWeight = weight;
        }
        groupId = gid;
        groupSize = 1;
      }
    }
    // handle the last group
    addOrUpdateInfos(prevTailGid, groupId, groupSize, curWeight, hasWeight, groupInfo, weightInfo);
    return groupId;
  }

  private static void addOrUpdateInfos(int prevTailGid, int curGid, int curGroupSize,
      Float curWeight, boolean hasWeight, List<Integer> groupInfo, List<Float> weightInfo) {
    if (curGroupSize <= 0) {
      return;
    }
    if (groupInfo.isEmpty() || curGid != prevTailGid) {
      // The first group of the first batch or a completely new group
      groupInfo.add(curGroupSize);

      // Weight: Add weight info
      if (hasWeight && curWeight != null) {
        weightInfo.add(curWeight);
      }
    } else {
      // This is the case when some rows at the beginning of this batch belong to
      // last group in previous batch, so update the group size for previous group info.
      groupInfo.set(groupInfo.size() - 1, curGroupSize);

      // No need to update the weight of last group since all the weights in a group are the same
    }
  }

  public static DType getRapidsType(DataType type) {
    DType result = toRapidsOrNull(type);
    if (result == null) {
      throw new IllegalArgumentException(type + " is not supported for GPU processing yet.");
    }
    return result;
  }

  private static DType toRapidsOrNull(DataType type) {
    if (type instanceof LongType) {
      return DType.INT64;
    } else if (type instanceof DoubleType) {
      return DType.FLOAT64;
    } else if (type instanceof ByteType) {
      return DType.INT8;
    } else if (type instanceof BooleanType) {
      return DType.BOOL8;
    } else if (type instanceof ShortType) {
      return DType.INT16;
    } else if (type instanceof IntegerType) {
      return DType.INT32;
    } else if (type instanceof FloatType) {
      return DType.FLOAT32;
    } else if (type instanceof DateType) {
      return DType.DATE32;
    } else if (type instanceof TimestampType) {
      return DType.TIMESTAMP;
    } else if (type instanceof StringType) {
      return DType.STRING; // TODO what do we want to do about STRING_CATEGORY???
    }
    return null;
  }
}
