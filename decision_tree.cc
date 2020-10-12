#include "decision_tree.h"

#include "bounding_box.h"
#include "interval.h"
#include "nlohmann/json.hpp"

using nlohmann::json;

namespace cz {

DecisionTree::DecisionTree(double label, int class_id)
    : is_leaf_(true), is_root_(false), label_(label), class_id_(class_id) {}

DecisionTree::~DecisionTree() {}

std::unique_ptr<DecisionTree> DecisionTree::CreateFromJson(const json& tree_obj,
                                                           int class_id,
                                                           bool is_root) {
  std::unique_ptr<DecisionTree> tree;
  if (tree_obj.contains("leaf")) {
    tree = std::make_unique<DecisionTree>(tree_obj["leaf"], class_id);
  } else {
    tree = std::make_unique<DecisionTree>(-1, class_id);
    tree->is_leaf_ = false;

    // tree_obj["split"] may be an integer or a string of index like "f12".
    if (tree_obj["split"].is_string()) {
      std::string s = tree_obj["split"];
      s = s.substr(1);
      tree->split_feature_id_ = std::stoi(s);
    } else {
      tree->split_feature_id_ = tree_obj["split"];
    }

    tree->split_condition_ = tree_obj["split_condition"];
    if (tree_obj["yes"] == tree_obj["children"][0]["nodeid"]) {
      assert(tree_obj["no"] == tree_obj["children"][1]["nodeid"]);
      tree->left_child_ =
          CreateFromJson(tree_obj["children"][0], class_id, false);
      tree->right_child_ =
          CreateFromJson(tree_obj["children"][1], class_id, false);
    } else {
      assert(tree_obj["yes"] == tree_obj["children"][1]["nodeid"]);
      assert(tree_obj["no"] == tree_obj["children"][0]["nodeid"]);
      tree->left_child_ =
          CreateFromJson(tree_obj["children"][1], class_id, false);
      tree->right_child_ =
          CreateFromJson(tree_obj["children"][0], class_id, false);
    }
  }

  tree->is_root_ = is_root;

  return std::move(tree);
}

int DecisionTree::ClassId() const {
  return class_id_;
}

double DecisionTree::PredictLabel(const Point& x) const {
  return FindPredictionNode(x)->label_;
}

void DecisionTree::ComputeBoundingBox() {
  // Root node.
  if (!box_)
    box_ = std::make_unique<BoundingBox>(this);

  if (left_child_) {
    left_child_->box_ = std::make_unique<BoundingBox>(*box_);
    left_child_->box_->IntersectFeature(split_feature_id_,
                                        Interval::Upper(split_condition_));
    left_child_->ComputeBoundingBox();
  }

  if (right_child_) {
    right_child_->box_ = std::make_unique<BoundingBox>(*box_);
    right_child_->box_->IntersectFeature(split_feature_id_,
                                         Interval::Lower(split_condition_));
    right_child_->ComputeBoundingBox();
  }

  if (is_leaf_) {
    box_->SetLabel(label_);
  }
}

BoundingBox* DecisionTree::GetBoundingBox(const Point& x) const {
  BoundingBox* b = FindPredictionNode(x)->box_.get();
  assert(b);
  return b;
}

void DecisionTree::FillFeatureSplits(
    std::vector<std::set<double>>* feature_splits) {
  if (is_leaf_)
    return;

  (*feature_splits)[split_feature_id_].insert(split_condition_);
  left_child_->FillFeatureSplits(feature_splits);
  right_child_->FillFeatureSplits(feature_splits);

  if (is_root_) {
    interesting_features_ = std::make_unique<std::vector<int>>();
    FillInterestingFeatures(interesting_features_.get());

    auto it = std::unique(interesting_features_->begin(),
                          interesting_features_->end());
    interesting_features_->resize(
        std::distance(interesting_features_->begin(), it));
  }
}

void DecisionTree::FillInterestingFeatures(
    std::vector<int>* interesting_features) const {
  if (is_leaf_)
    return;

  interesting_features->push_back(split_feature_id_);
  left_child_->FillInterestingFeatures(interesting_features);
  right_child_->FillInterestingFeatures(interesting_features);
}

std::vector<const BoundingBox*> DecisionTree::GetAlternativeNodes(
    const BoundingBox& relaxed_box) const {
  std::vector<const BoundingBox*> nodes;

  GetAlternativeNodes(relaxed_box, &nodes);
  return std::move(nodes);
}

void DecisionTree::GetAlternativeNodes(
    const BoundingBox& relaxed_box,
    std::vector<const BoundingBox*>* nodes) const {
  if (is_leaf_) {
    nodes->push_back(box_.get());
    return;
  }

  const auto& relaxed_bound = relaxed_box.GetOrEmpty(split_feature_id_);

  if (left_child_) {
    Interval left_interval = Interval::Upper(split_condition_);
    left_interval.Intersect(relaxed_bound);
    if (left_interval.HasValue())
      left_child_->GetAlternativeNodes(relaxed_box, nodes);
  }

  if (right_child_) {
    Interval right_interval = Interval::Lower(split_condition_);
    right_interval.Intersect(relaxed_bound);
    if (right_interval.HasValue())
      right_child_->GetAlternativeNodes(relaxed_box, nodes);
  }
}

std::vector<const BoundingBox*> DecisionTree::GetLeaves() const {
  std::vector<const BoundingBox*> leaves;
  CDfs([&](const DecisionTree* t) -> std::pair<bool, bool> {
    if (t->is_leaf()) {
      leaves.push_back(t->box());
      return {false, false};
    }
    return {true, true};
  });
  return std::move(leaves);
}

const std::vector<int>& DecisionTree::GetInterestngFeatures() const {
  return *interesting_features_;
}

void DecisionTree::Dfs(const DfsFunc& f) {
  auto left_right = f(this);
  if (is_leaf_)
    return;

  if (left_right.first)
    left_child_->Dfs(f);

  if (left_right.second)
    right_child_->Dfs(f);
}

void DecisionTree::CDfs(const CDfsFunc& f) const {
  auto left_right = f(this);
  if (is_leaf_)
    return;

  if (left_right.first)
    left_child_->CDfs(f);

  if (left_right.second)
    right_child_->CDfs(f);
}

void DecisionTree::SetSplitCondition(int split_feature_id,
                                     double split_condition,
                                     double left_label,
                                     double right_label) {
  assert(is_leaf_);
  is_leaf_ = false;
  split_feature_id_ = split_feature_id;
  split_condition_ = split_condition;

  left_child_ = std::make_unique<DecisionTree>(left_label, class_id_);
  right_child_ = std::make_unique<DecisionTree>(right_label, class_id_);
}

const DecisionTree* DecisionTree::FindPredictionNode(const Point& x) const {
  if (is_leaf_)
    return this;
  if (x[split_feature_id_] < split_condition_)
    return left_child_->FindPredictionNode(x);
  return right_child_->FindPredictionNode(x);
}

}  // namespace cz
