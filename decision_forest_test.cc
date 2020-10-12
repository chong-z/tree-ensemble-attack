#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>

#include "bounding_box.h"
#include "decision_forest.h"
#include "decision_tree.h"
#include "test.h"
#include "utility.h"

namespace cz {

class DecisionForestTest {
 public:
  // Decision Tree
  static std::unique_ptr<DecisionTree> CreateTestTree1() {
    auto tree = std::make_unique<DecisionTree>(0, 1);
    tree->SetSplitCondition(0, 10, -1, 1);
    tree->left_child_->SetSplitCondition(1, 10, -2, 2);
    tree->right_child_->SetSplitCondition(2, 10, -3, 3);
    return std::move(tree);
  }

  static std::unique_ptr<DecisionTree> CreateTestTree2() {
    auto tree = std::make_unique<DecisionTree>(0, 1);
    tree->SetSplitCondition(0, 5, -1, 1);
    tree->left_child_->SetSplitCondition(1, 2, -4, 4);
    return std::move(tree);
  }

  // Decision Forest
  static std::unique_ptr<DecisionForest> CreateTestForest() {
    auto forest = std::make_unique<DecisionForest>(2, 2);
    forest->AddDecisionTree(CreateTestTree1());
    forest->AddDecisionTree(CreateTestTree2());
    forest->Setup();
    return std::move(forest);
  }

  static std::unique_ptr<DecisionForest> CreateDoubleTestForest() {
    auto forest = std::make_unique<DecisionForest>(2, 2);
    forest->AddDecisionTree(CreateTestTree1());
    forest->AddDecisionTree(CreateTestTree2());
    forest->AddDecisionTree(CreateTestTree1());
    forest->AddDecisionTree(CreateTestTree2());
    forest->Setup();
    return std::move(forest);
  }
};

void DecisionTreeBasicTest() {
  auto tree = DecisionForestTest::CreateTestTree1();
  EXPECT_EQ(tree->PredictLabel({0, 0, 0}), -2.0);
  EXPECT_EQ(tree->PredictLabel({0, 20, 0}), 2.0);
  EXPECT_EQ(tree->PredictLabel({20, 0, 0}), -3.0);
  EXPECT_EQ(tree->PredictLabel({20, 0, 20}), 3.0);
}

void DecisionTreeBoundingBoxTest() {
  auto tree = DecisionForestTest::CreateTestTree1();
  tree->ComputeBoundingBox();
  EXPECT_EQ(tree->GetBoundingBox({0, 0, 0})->ToDebugString(),
            std::string("[0:(INF,10.000000),1:(INF,10.000000),]"));
  EXPECT_EQ(tree->GetBoundingBox({0, 11, 0})->ToDebugString(),
            std::string("[0:(INF,10.000000),1:(10.000000,INF),]"));
  EXPECT_EQ(tree->GetBoundingBox({11, 11, 11})->ToDebugString(),
            std::string("[0:(10.000000,INF),2:(10.000000,INF),]"));
}

void DecisionForestBasicTest() {
  auto forest = DecisionForestTest::CreateTestForest();
  EXPECT_EQ(forest->ComputeBinaryScoreForTesting({0, 0, 0}), -6.0);
  EXPECT_EQ(forest->ComputeBinaryScoreForTesting({0, 20, 0}), 6.0);
  EXPECT_EQ(forest->ComputeBinaryScoreForTesting({20, 0, 0}), -2.0);
  EXPECT_EQ(forest->ComputeBinaryScoreForTesting({20, 0, 20}), 4.0);
}

void DecisionForestBoundingBoxTest() {
  auto forest = DecisionForestTest::CreateTestForest();
  EXPECT_EQ(forest->GetBoundingBox({0, 0, 0}).ToDebugString(),
            std::string("[0:(INF,5.000000),1:(INF,2.000000),]"));
  EXPECT_EQ(forest->GetBoundingBox({4, 2, 0}).ToDebugString(),
            std::string("[0:(INF,5.000000),1:(2.000000,10.000000),]"));
  EXPECT_EQ(forest->GetBoundingBox({5, 2, 0}).ToDebugString(),
            std::string("[0:(5.000000,10.000000),1:(INF,10.000000),]"));
  EXPECT_EQ(forest->GetBoundingBox({0, 11, 0}).ToDebugString(),
            std::string("[0:(INF,5.000000),1:(10.000000,INF),]"));
  EXPECT_EQ(forest->GetBoundingBox({11, 11, 11}).ToDebugString(),
            std::string("[0:(10.000000,INF),2:(10.000000,INF),]"));
}

void DecisionForestLayeredBoundingBoxSingleTest() {
  auto forest = DecisionForestTest::CreateTestForest();
  auto layered_box = forest->GetLayeredBoundingBox({5, 2, 0});
  layered_box->VerifyCachedIntersectionForTesting();
  EXPECT_EQ(layered_box->GetCachedIntersection()->ToDebugString(),
            std::string("[0:(5.000000,10.000000),1:(INF,10.000000),]"));

  auto indenpendent_boxes = layered_box->GetIndenpendentBoundingBoxes();
  EXPECT_EQ(int(indenpendent_boxes.size()), 2);
  std::vector<std::string> box_s;
  for (const auto& box : indenpendent_boxes) {
    box_s.push_back(box.ToDebugString());
  }
  std::sort(box_s.begin(), box_s.end());
  EXPECT_EQ(box_s[0], std::string("[0:(5.000000,INF),]"));
  EXPECT_EQ(box_s[1], std::string("[0:(INF,10.000000),1:(INF,10.000000),]"));
  layered_box->ShiftPoint({4, 2, 0});
  layered_box->VerifyCachedIntersectionForTesting();
  EXPECT_EQ(layered_box->GetCachedIntersection()->ToDebugString(),
            std::string("[0:(INF,5.000000),1:(2.000000,10.000000),]"));
}

void DecisionForestLayeredBoundingBoxDoubleTest() {
  auto forest = DecisionForestTest::CreateDoubleTestForest();
  auto layered_box = forest->GetLayeredBoundingBox({5, 2, 0});
  layered_box->VerifyCachedIntersectionForTesting();
  EXPECT_EQ(layered_box->GetCachedIntersection()->ToDebugString(),
            std::string("[0:(5.000000,10.000000),1:(INF,10.000000),]"));

  auto indenpendent_boxes = layered_box->GetIndenpendentBoundingBoxes();
  EXPECT_EQ(int(indenpendent_boxes.size()), 4);
  std::vector<std::string> box_s;
  for (const auto& box : indenpendent_boxes) {
    box_s.push_back(box.ToDebugString());
  }
  std::sort(box_s.begin(), box_s.end());
  EXPECT_EQ(box_s[0], std::string("[0:(5.000000,INF),]"));
  EXPECT_EQ(box_s[1], std::string("[0:(5.000000,INF),]"));
  EXPECT_EQ(box_s[2], std::string("[0:(INF,10.000000),1:(INF,10.000000),]"));
  EXPECT_EQ(box_s[3], std::string("[0:(INF,10.000000),1:(INF,10.000000),]"));
  layered_box->ShiftPoint({4, 2, 0});
  layered_box->VerifyCachedIntersectionForTesting();
  EXPECT_EQ(layered_box->GetCachedIntersection()->ToDebugString(),
            std::string("[0:(INF,5.000000),1:(2.000000,10.000000),]"));
}

void DecisionForestJsonTest() {
  auto forest = DecisionForest::CreateFromJson(
      "testing/breast_cancer_robust.0008.json", 2, 10);
  EXPECT_EQ(forest->ComputeBinaryScoreForTesting(
                {0, 0, 0, 0, 0.3, 0, 0.4, 0.38, 0, 0.45, 0.38}),
            -0.4680399897);
}

void DecisionForestMulticlassTest() {
  auto forest =
      DecisionForest::CreateFromJson("testing/fashion_robust.json", 10, 784);
  auto y_X_list = LoadSVMFile("testing/small-fashion-mnist.libsvm", 784, 1);
  int num_correct = 0;
  for (const auto& y_X : y_X_list) {
    int predict_y = forest->PredictLabel(y_X.second);
    if (predict_y == y_X.first)
      ++num_correct;
  }
  EXPECT_EQ((int)y_X_list.size(), 30);
  EXPECT_EQ(num_correct, 30);
}

}  // namespace cz

int main() {
  srand(0);
  RUN_TEST(cz::DecisionTreeBasicTest);
  RUN_TEST(cz::DecisionTreeBoundingBoxTest);
  RUN_TEST(cz::DecisionForestBasicTest);
  RUN_TEST(cz::DecisionForestBoundingBoxTest);
  RUN_TEST(cz::DecisionForestLayeredBoundingBoxSingleTest);
  RUN_TEST(cz::DecisionForestLayeredBoundingBoxDoubleTest);
  RUN_TEST(cz::DecisionForestJsonTest);
  RUN_TEST(cz::DecisionForestMulticlassTest);
}
