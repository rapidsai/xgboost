// Copyright by Contributors
#include <xgboost/objective.h>
#include <xgboost/generic_parameters.h>
#include "../helpers.h"

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);

  xgboost::ObjFunction * obj =
      xgboost::ObjFunction::Create("rank:pairwise", &lparam);
  obj->Configure(args);
  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                   {0, 0.1f, 0, 0.1f},
                   {0,   1, 0, 1},
                   {2.0f, 0.0f},
                   {0, 2, 4},
                   {1.9f, -1.9f, 0.0f, 0.0f},
                   {1.995f, 1.995f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                   {0, 0.1f, 0, 0.1f},
                   {0,   1, 0, 1},
                   {1.0f, 1.0f},
                   {0, 2, 4},
                   {0.95f, -0.95f,  0.95f, -0.95f},
                   {0.9975f, 0.9975f, 0.9975f, 0.9975f});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());

  delete obj;
}

TEST(Objective, DeclareUnifiedTest(PairwiseRankingGPairSameLabels)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);

  xgboost::ObjFunction *obj =
      xgboost::ObjFunction::Create("rank:pairwise", &lparam);
  obj->Configure(args);
  // No computation of gradient/hessian, as there is no diversity in labels
  CheckRankingObjFunction(obj,
                   {0, 0.1f, 0, 0.1f},
                   {1,   1, 1, 1},
                   {2.0f, 0.0f},
                   {0, 2, 4},
                   {0.0f, 0.0f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 0.0f, 0.0f});

  ASSERT_NO_THROW(obj->DefaultEvalMetric());

  delete obj;
}

TEST(Objective, DeclareUnifiedTest(NDCGRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);

  xgboost::ObjFunction *obj =
    xgboost::ObjFunction::Create("rank:ndcg", &lparam);
  obj->Configure(args);

  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {0.7f, -0.7f, 0.0f, 0.0f},
                          {0.74f, 0.74f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {1.0f, 1.0f},
                          {0, 2, 4},
                          {0.35f, -0.35f,  0.35f, -0.35f},
                          {0.368f, 0.368f, 0.368f, 0.368f});
  ASSERT_NO_THROW(obj->DefaultEvalMetric());

  delete obj;
}

TEST(Objective, DeclareUnifiedTest(MAPRankingGPair)) {
  std::vector<std::pair<std::string, std::string>> args;
  xgboost::GenericParameter lparam = xgboost::CreateEmptyGenericParam(0, NGPUS);

  xgboost::ObjFunction *obj =
    xgboost::ObjFunction::Create("rank:map", &lparam);
  obj->Configure(args);

  // Test with setting sample weight to second query group
  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {2.0f, 0.0f},
                          {0, 2, 4},
                          {0.95f, -0.95f,  0.0f, 0.0f},
                          {0.9975f, 0.9975f, 0.0f, 0.0f});

  CheckRankingObjFunction(obj,
                          {0, 0.1f, 0, 0.1f},
                          {0,   1, 0, 1},
                          {1.0f, 1.0f},
                          {0, 2, 4},
                          {0.475f, -0.475f,  0.475f, -0.475f},
                          {0.4988f, 0.4988f, 0.4988f, 0.4988f});
  ASSERT_NO_THROW(obj->DefaultEvalMetric());

  delete obj;
}
