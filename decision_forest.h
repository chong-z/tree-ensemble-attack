#pragma once

#include <string>

#include "utility.h"

namespace cz {

class BoundingBox;
class LayeredBoundingBox;
class DecisionTree;

class DecisionForest {
 public:
  DecisionForest(int num_class, int max_feature_id);
  ~DecisionForest();

  static std::unique_ptr<DecisionForest> CreateFromJson(const std::string& path,
                                                        int num_class,
                                                        int max_feature_id);

  int PredictLabel(const Point& x) const;
  int PredictLabelBetween(const Point& x, int class1, int class2) const;

  std::unique_ptr<LayeredBoundingBox>
  GetLayeredBoundingBox(const Point& x, int class1 = -1, int class2 = -1) const;

  BoundingBox GetBoundingBox(const Point& x) const;

  const std::vector<std::vector<double>>& FeatureSplits() const;

  int HammingDistanceBetween(const Point& p1, const Point& p2, int class1, int class2) const;

  std::vector<double> ComputeScores(const Point& x) const;

  double ComputeBinaryScoreForTesting(const Point& x) const;

  int NumTreesForTesting() const;

 private:
  friend class NeighborAttack;
  friend class DecisionForestTest;
  friend class NeighborAttackTest;

  void Setup();

  void ComputeBoundingBox();
  void ComputeFeatureSplits();
  void AddDecisionTree(std::unique_ptr<DecisionTree> tree);

  bool has_bounding_box_ = false;
  std::vector<std::unique_ptr<DecisionTree>> trees_;

  // <feature_id, sorted_splits>.
  std::unique_ptr<std::vector<std::vector<double>>> feature_splits_;

  int num_class_;
  int max_feature_id_;

  DISALLOW_COPY_AND_ASSIGN(DecisionForest);
};

}  // namespace cz
