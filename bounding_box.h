#pragma once

#include <map>
#include <set>
#include <string>

#include <boost/container/flat_map.hpp>

#include "interval.h"
#include "utility.h"

namespace cz {

class DecisionTree;
class DecisionForest;
class OrderedBoxes;

class BoundingBox {
 public:
  using IntervalsType = boost::container::flat_map<int, Interval>;

  BoundingBox();
  explicit BoundingBox(const DecisionTree*);
  BoundingBox(BoundingBox&&);
  BoundingBox(const BoundingBox&) = default;
  ~BoundingBox();

  void IntersectFeature(int feature_id, const Interval&);
  void Intersect(const BoundingBox&);
  bool Overlaps(const BoundingBox&);

  bool HasUpper(int feature_id) const;
  double Upper(int feature_id) const;

  bool HasLower(int feature_id) const;
  double Lower(int feature_id) const;

  bool Contains(const Point&) const;

  Patch ClosestPatchTo(const Point& point) const;
  double NormTo(const Point& point, int norm_type) const;

  IntervalsType& Intervals();
  const IntervalsType& Intervals() const;

  Interval& operator[](int feature_id);
  const Interval& operator[](int feature_id) const;

  Interval GetOrEmpty(int feature_id) const;

  bool operator==(const BoundingBox&) const;

  const DecisionTree* OwnerTree() const;

  double Label() const;
  void SetLabel(double label);

  std::string ToDebugString() const;

  void Clear();

 private:
  // feature_id -> Interval.
  IntervalsType intervals_;

  const DecisionTree* owner_tree_ = nullptr;

  // Prediction label for this box.
  double label_ = 0;
};

class LayeredBoundingBox {
 public:
  LayeredBoundingBox(const DecisionForest* owner_forest,
                     int num_class,
                     int max_feature_id,
                     int class1,
                     int class2);
  ~LayeredBoundingBox();

  void AddBox(const BoundingBox*);

  int PredictionLabel(const std::vector<double>* scores = nullptr) const;
  double LabelScore(int victim_label,
                    const std::vector<double>* scores = nullptr) const;
  const std::vector<double>& Scores() const;

  std::vector<const BoundingBox*> GetEffectiveBoxesForFeature(
      int feature_id,
      SearchMode search_mode) const;

  std::vector<FeatureDir> GetBoundedFeatures() const;

  std::vector<const BoundingBox*> GetAlternativeBoxes(
      const BoundingBox& target_feature_constrain,
      int max_dist,
      const BoundingBox* box_to_replace,
      bool enable_relaxed_boundary,
      const BoundingBox* hard_constrain = nullptr) const;

  void FillIncompatibleBoxes(
      int feature_id,
      double value,
      std::vector<const BoundingBox*>* incompatible_boxes) const;

  // Current |location_| must also be the optimized point.
  Patch StretchWithinBox(
      const Patch& patch,
      const Point& victim_point,
      const BoundingBox* constrain_box,
      const std::vector<const BoundingBox*>& incompatible_boxes) const;

  std::vector<const BoundingBox*> GetNewBoxes(
      const Patch& patch,
      const std::vector<const BoundingBox*>& incompatible_boxes) const;

  std::vector<double> GetNewScores(
      const std::vector<const BoundingBox*>& incompatible_boxes,
      const std::vector<const BoundingBox*>& new_boxes) const;

  void TightenPoint(Point* new_adv,
                    const std::vector<const BoundingBox*>& new_boxes) const;

  void ShiftPoint(const Point& point);
  // |ShiftByPatch| is more efficient.
  void ShiftByPatch(const Patch& patch);
  void ShiftByDirection(const Direction& dir);
  const BoundingBox* GetCachedIntersection() const;

  std::vector<BoundingBox> GetIndenpendentBoundingBoxes() const;

  void SetInitialLocation(const Point& initial_location);
  const Point& Location() const;

  void VerifyCachedIntersectionForTesting() const;

  size_t Hash() const;

  const BoundingBox* GetBoxForTree(const DecisionTree*) const;
  std::vector<const BoundingBox*> GetBoxForAllTree() const;

  bool CheckScoresForTesting(const Patch& patch,
                             const std::vector<double>& scores) const;
  void AssertTightForTesting(const Point& victim_point) const;

 private:
  // Won't update |ordered_boxes_|.
  void RemoveBox(const BoundingBox* box);

  Point location_;
  int class1_ = -1;
  int class2_ = -1;

  std::hash<const BoundingBox*> ptr_hasher_;
  size_t hash_ = 0;

  std::unique_ptr<OrderedBoxes> ordered_boxes_;

  // One box per tree.
  std::map<const DecisionTree*, const BoundingBox*> boxes_;

  // Scores per class.
  std::vector<double> scores_;

  const DecisionForest* owner_forest_ = nullptr;

  DISALLOW_COPY_AND_ASSIGN(LayeredBoundingBox);
};

}  // namespace cz
