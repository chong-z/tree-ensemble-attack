#include "decision_forest.h"

#include <fstream>

#include "bounding_box.h"
#include "decision_tree.h"
#include "nlohmann/json.hpp"

using nlohmann::json;

namespace cz {

DecisionForest::DecisionForest(int num_class, int max_feature_id)
    : num_class_(num_class), max_feature_id_(max_feature_id) {}
DecisionForest::~DecisionForest() {}

std::unique_ptr<DecisionForest> DecisionForest::CreateFromJson(
    const std::string& path,
    int num_class,
    int max_feature_id) {
  std::ifstream fin(path);
  json forest_array;
  fin >> forest_array;

  auto forest = std::make_unique<DecisionForest>(num_class, max_feature_id);
  int tree_count = 0;
  for (const auto& tree_obj : forest_array) {
    assert(tree_obj.is_object());
    int class_id = (num_class == 2) ? 1 : (tree_count % num_class);
    forest->AddDecisionTree(
        DecisionTree::CreateFromJson(tree_obj, class_id, true));
    ++tree_count;
  }

  forest->Setup();

  return std::move(forest);
}

int DecisionForest::PredictLabel(const Point& x) const {
  return MaxIndex(ComputeScores(x));
}

int DecisionForest::PredictLabelBetween(const Point& x,
                                        int class1,
                                        int class2) const {
  double score = 0;
  for (const auto& tree : trees_) {
    if (tree->ClassId() == class2) {
      score += tree->PredictLabel(x);
    } else if (tree->ClassId() == class1) {
      score -= tree->PredictLabel(x);
    }
  }
  if (score > 0)
    return class2;
  return class1;
}

void DecisionForest::Setup() {
  ComputeBoundingBox();
  ComputeFeatureSplits();
}

void DecisionForest::ComputeBoundingBox() {
  assert(!has_bounding_box_);
  for (auto& t : trees_)
    t->ComputeBoundingBox();
  has_bounding_box_ = true;
}

void DecisionForest::ComputeFeatureSplits() {
  assert(!feature_splits_);
  feature_splits_ =
      std::make_unique<std::vector<std::vector<double>>>(max_feature_id_ + 1);
  std::vector<std::set<double>> feature_splits_set(max_feature_id_ + 1);
  for (const auto& t : trees_)
    t->FillFeatureSplits(&feature_splits_set);
  for (int i = 0; i <= max_feature_id_; ++i) {
    (*feature_splits_)[i].push_back(-0.1);
    (*feature_splits_)[i].insert((*feature_splits_)[i].end(),
                                 feature_splits_set[i].begin(),
                                 feature_splits_set[i].end());
    (*feature_splits_)[i].push_back(1.1);
  }
}

std::vector<double> DecisionForest::ComputeScores(const Point& x) const {
  std::vector<double> scores(num_class_, 0);
  assert(num_class_ >= 2);
  for (const auto& tree : trees_)
    scores[tree->ClassId()] += tree->PredictLabel(x);
  return std::move(scores);
}

std::unique_ptr<LayeredBoundingBox> DecisionForest::GetLayeredBoundingBox(
    const Point& x,
    int class1,
    int class2) const {
  assert(has_bounding_box_);

  auto box = std::make_unique<LayeredBoundingBox>(
      this, num_class_, max_feature_id_, class1, class2);
  box->SetInitialLocation(x);
  for (const auto& t : trees_) {
    if (t->ClassId() == class1 || t->ClassId() == class2 ||
        (class1 == -1 && class2 == -1)) {
      box->AddBox(t->GetBoundingBox(x));
    }
  }
  return std::move(box);
}

BoundingBox DecisionForest::GetBoundingBox(const Point& x) const {
  BoundingBox joint_box;
  for (const auto& t : trees_) {
    joint_box.Intersect(*t->GetBoundingBox(x));
  }
  return std::move(joint_box);
}

const std::vector<std::vector<double>>& DecisionForest::FeatureSplits() const {
  return *feature_splits_;
}

int DecisionForest::HammingDistanceBetween(const Point& p1, const Point& p2, int class1, int class2) const {
  int dist = 0;
  for (const auto& t : trees_) {
    if (t->ClassId() != class1 && t->ClassId() != class2)
      continue;

    if (t->GetBoundingBox(p1) != t->GetBoundingBox(p2))
      ++dist;
  }
  return dist;
}

double DecisionForest::ComputeBinaryScoreForTesting(const Point& x) const {
  assert(num_class_ == 2);
  return ComputeScores(x)[1];
}

int DecisionForest::NumTreesForTesting() const {
  return trees_.size();
}

void DecisionForest::AddDecisionTree(std::unique_ptr<DecisionTree> tree) {
  trees_.push_back(std::move(tree));
}

}  // namespace cz
