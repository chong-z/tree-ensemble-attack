#pragma once

#include "nlohmann/json_fwd.hpp"
#include "utility.h"

namespace cz {

class BoundingBox;
class LayeredBoundingBox;

class DecisionTree {
 public:
  DecisionTree(double label, int class_id);
  ~DecisionTree();

  static std::unique_ptr<DecisionTree>
  CreateFromJson(const nlohmann::json& tree_obj, int class_id, bool is_root);

  int ClassId() const;
  double PredictLabel(const Point& x) const;

  // Must call |ComputeBoundingBox| prior to |GetBoundingBox|.
  void ComputeBoundingBox();
  BoundingBox* GetBoundingBox(const Point& x) const;

  void FillFeatureSplits(std::vector<std::set<double>>* feature_splits);

  std::vector<const BoundingBox*> GetAlternativeNodes(
      const BoundingBox& relaxed_box) const;

  std::vector<const BoundingBox*> GetLeaves() const;

  // Returns all features used as |split_feature_id_|.
  const std::vector<int>& GetInterestngFeatures() const;

  // <dfs_left, dfs_right>(tree);
  using DfsFunc = std::function<std::pair<bool, bool>(DecisionTree*)>;
  void Dfs(const DfsFunc&);

  using CDfsFunc = std::function<std::pair<bool, bool>(const DecisionTree*)>;
  void CDfs(const CDfsFunc&) const;

  auto split_feature_id() const { return split_feature_id_; }
  auto split_condition() const { return split_condition_; }
  auto is_leaf() const { return is_leaf_; }
  const auto* box() const { return box_.get(); }

 private:
  friend class DecisionForestTest;
  friend class NeighborAttackTest;

  void SetSplitCondition(int split_feature_id,
                         double split_condition,
                         double left_label,
                         double right_label);

  void FillInterestingFeatures(std::vector<int>* interesting_features) const;

  const DecisionTree* FindPredictionNode(const Point& x) const;

  void GetAlternativeNodes(const BoundingBox& relaxed_box,
                           std::vector<const BoundingBox*>* nodes) const;

  bool is_leaf_;
  bool is_root_;
  int class_id_;
  double label_;

  int split_feature_id_;
  double split_condition_;

  // For root node only.
  std::unique_ptr<std::vector<int>> interesting_features_;

  // "Yes" subtree, |point[split_feature_id_] < split_condition_|.
  std::unique_ptr<DecisionTree> left_child_;
  // "No" subtree, |point[split_feature_id_] >= split_condition_|.
  std::unique_ptr<DecisionTree> right_child_;

  std::unique_ptr<BoundingBox> box_;

  DISALLOW_COPY_AND_ASSIGN(DecisionTree);
};

}  // namespace cz
