#include "bounding_box.h"

#include <list>
#include <unordered_map>

#include "decision_forest.h"
#include "decision_tree.h"
#include "timing.h"

namespace cz {

using LowerCMP = std::greater<double>;
using UpperCMP = std::less<double>;

class OrderedBoxes {
 public:
  explicit OrderedBoxes(int max_feature_id)
      : lower_(max_feature_id + 1), upper_(max_feature_id + 1) {}

  const BoundingBox* GetCachedIntersection() const {
    return &cached_intersection_;
  }

  void Add(const BoundingBox* box) {
    Timing::Instance()->StartTimer("OrderedBoxes::Add");
    Timing::Instance()->BinCount("OrderedBoxes::Add::Intervals",
                                 box->Intervals().size());
    for (const auto& feature_interval : box->Intervals()) {
      int feature_id = feature_interval.first;
      const auto& interval = feature_interval.second;

      if (interval.lower.has_value())
        lower_[feature_id].insert(std::make_pair(interval.lower.value(), box));

      if (interval.upper.has_value())
        upper_[feature_id].insert(std::make_pair(interval.upper.value(), box));
    }

    Timing::Instance()->StartTimer(
        "OrderedBoxes::Add::cached_intersection_.Intersect");
    cached_intersection_.Intersect(*box);
    Timing::Instance()->EndTimer(
        "OrderedBoxes::Add::cached_intersection_.Intersect");

    Timing::Instance()->EndTimer("OrderedBoxes::Add");
  }

  std::vector<const BoundingBox*> RemoveUntil(int feature_id,
                                              double value,
                                              bool is_upper) {
    Timing::Instance()->StartTimer("OrderedBoxes::RemoveUntil");

    std::vector<const BoundingBox*> boxes;
    FillIncompatibleBoxes(feature_id, value, is_upper, &boxes);
    Remove(boxes);

    Timing::Instance()->EndTimer("OrderedBoxes::RemoveUntil");
    return std::move(boxes);
  }

  void FillIncompatibleBoxes(
      int feature_id,
      double value,
      bool is_upper,
      std::vector<const BoundingBox*>* incompatible_boxes,
      std::optional<double>* new_bound = nullptr) const {
    Timing::Instance()->StartTimer("OrderedBoxes::FillIncompatibleBoxes");
    if (is_upper) {
      FillIncompatibleBoxesUpper(feature_id, value, incompatible_boxes,
                                 new_bound);
    } else {
      FillIncompatibleBoxesLower(feature_id, value, incompatible_boxes,
                                 new_bound);
    }
    Timing::Instance()->EndTimer("OrderedBoxes::FillIncompatibleBoxes");
  }

  std::optional<double> GetUpperExcept(
      int feature_id,
      const std::vector<const BoundingBox*>& except_box_sorted) {
    auto iter = upper_[feature_id].cbegin();
    auto end = upper_[feature_id].cend();
    while (iter != end &&
           std::binary_search(except_box_sorted.begin(),
                              except_box_sorted.end(), iter->second))
      ++iter;

    if (iter != end)
      return iter->first;
    return std::nullopt;
  }

  std::optional<double> GetLowerExcept(
      int feature_id,
      const std::vector<const BoundingBox*>& except_box_sorted) {
    auto iter = lower_[feature_id].cbegin();
    auto end = lower_[feature_id].cend();
    while (iter != end &&
           std::binary_search(except_box_sorted.begin(),
                              except_box_sorted.end(), iter->second))
      ++iter;

    if (iter != end)
      return iter->first;
    return std::nullopt;
  }

  template <class CMP>
  std::optional<double> GetKthBound(int feature_id,
                                    int max_dist,
                                    std::optional<double> filter) const {
    auto iter = Begin<CMP>(feature_id);
    auto end = End<CMP>(feature_id);
    int dist = 1;
    while (iter != end) {
      if (filter.has_value() && iter->first == filter.value()) {
        iter = Bounds<CMP>()[feature_id].upper_bound(iter->first);
        continue;
      }

      if (dist < max_dist) {
        ++dist;
        iter = Bounds<CMP>()[feature_id].upper_bound(iter->first);
        continue;
      }

      break;
    }

    if (iter != end)
      return iter->first;
    return std::nullopt;
  }

  Interval GetKthInterval(int feature_id,
                          int max_dist,
                          const Interval& filter) const {
    // Timing::Instance()->StartTimer("GetKthInterval");
    Interval inter;
    inter.lower = GetKthBound<LowerCMP>(feature_id, max_dist, filter.lower);
    inter.upper = GetKthBound<UpperCMP>(feature_id, max_dist, filter.upper);
    // Timing::Instance()->EndTimer("GetKthInterval");
    return inter;
  }

 private:
  bool HasUpper(int feature_id) { return !upper_[feature_id].empty(); }
  double Upper(int feature_id) { return upper_[feature_id].begin()->first; }

  bool HasLower(int feature_id) { return !lower_[feature_id].empty(); }
  double Lower(int feature_id) { return lower_[feature_id].begin()->first; }

  template <class CMP>
  auto Begin(int feature_id) const {
    return std::cbegin(Bounds<CMP>()[feature_id]);
  }

  template <class CMP>
  auto End(int feature_id) const {
    return std::cend(Bounds<CMP>()[feature_id]);
  }

  template <class CMP>
  const auto& Bounds() const;

  void FillIncompatibleBoxesUpper(
      int feature_id,
      double value,
      std::vector<const BoundingBox*>* incompatible_boxes,
      std::optional<double>* new_upper = nullptr) const {
    auto iter = upper_[feature_id].cbegin();
    auto end = upper_[feature_id].cend();
    while (iter != end && value >= iter->first) {
      const auto* box = iter->second;
      incompatible_boxes->push_back(box);
      ++iter;
    }

    if (new_upper && iter != end)
      *new_upper = iter->first;
  }

  void FillIncompatibleBoxesLower(
      int feature_id,
      double value,
      std::vector<const BoundingBox*>* incompatible_boxes,
      std::optional<double>* new_lower = nullptr) const {
    auto iter = lower_[feature_id].cbegin();
    auto end = lower_[feature_id].cend();
    while (iter != end && value < iter->first) {
      const auto* box = iter->second;
      incompatible_boxes->push_back(box);
      ++iter;
    }

    if (new_lower && iter != end)
      *new_lower = iter->first;
  }

  void Remove(const BoundingBox* box) {
    for (const auto& feature_interval : box->Intervals()) {
      int feature_id = feature_interval.first;
      const auto& interval = feature_interval.second;

      auto cached = cached_intersection_[feature_id];

      if (interval.lower.has_value()) {
        const auto& end = lower_[feature_id].end();
        auto iter = lower_[feature_id].find(interval.lower.value());
        assert(iter != end);
        while (iter->second != box) {
          ++iter;
          assert(iter != end);
        }
        lower_[feature_id].erase(iter);

        if (interval.HasStricterLower(cached)) {
          if (HasLower(feature_id)) {
            cached.lower = Lower(feature_id);
          } else {
            cached.lower.reset();
          }
        }
      }

      if (interval.upper.has_value()) {
        const auto& end = upper_[feature_id].end();
        auto iter = upper_[feature_id].find(interval.upper.value());
        assert(iter != end);
        while (iter->second != box) {
          ++iter;
          assert(iter != end);
        }
        upper_[feature_id].erase(iter);

        if (interval.HasStricterUpper(cached)) {
          if (HasUpper(feature_id)) {
            cached.upper = Upper(feature_id);
          } else {
            cached.upper.reset();
          }
        }
      }

      if (cached.HasValue()) {
        cached_intersection_[feature_id] = cached;
      } else {
        cached_intersection_.Intervals().erase(feature_id);
      }
    }
  }

  void Remove(const std::vector<const BoundingBox*>& boxes) {
    Timing::Instance()->StartTimer("OrderedBoxes::Remove");
    for (const auto* box : boxes)
      Remove(box);
    Timing::Instance()->EndTimer("OrderedBoxes::Remove");
  }

  BoundingBox cached_intersection_;

  // <feature_id, <value, box>>.
  std::vector<std::multimap<double, const BoundingBox*, LowerCMP>> lower_;
  std::vector<std::multimap<double, const BoundingBox*, UpperCMP>> upper_;
};

template <>
const auto& OrderedBoxes::Bounds<LowerCMP>() const {
  return lower_;
}

template <>
const auto& OrderedBoxes::Bounds<UpperCMP>() const {
  return upper_;
}

// BoundingBox

BoundingBox::BoundingBox() {}
BoundingBox::BoundingBox(const DecisionTree* owner_tree)
    : owner_tree_(owner_tree) {}
BoundingBox::BoundingBox(BoundingBox&& rhs)
    : intervals_(std::move(rhs.intervals_)),
      owner_tree_(rhs.owner_tree_),
      label_(rhs.label_) {}
BoundingBox::~BoundingBox() {}

void BoundingBox::IntersectFeature(int feature_id, const Interval& interval) {
  intervals_[feature_id].Intersect(interval);
}

void BoundingBox::Intersect(const BoundingBox& other) {
  for (const auto& iter : other.intervals_)
    intervals_[iter.first].Intersect(iter.second);
}

bool BoundingBox::Overlaps(const BoundingBox& other) {
  for (const auto& iter : other.intervals_) {
    if (!intervals_[iter.first].Overlaps(iter.second))
      return false;
  }
  return true;
}

bool BoundingBox::HasUpper(int feature_id) const {
  auto iter = intervals_.find(feature_id);
  return iter != intervals_.end() && iter->second.upper.has_value();
}

double BoundingBox::Upper(int feature_id) const {
  return intervals_.find(feature_id)->second.upper.value();
}

bool BoundingBox::HasLower(int feature_id) const {
  auto iter = intervals_.find(feature_id);
  return iter != intervals_.end() && iter->second.lower.has_value();
}

double BoundingBox::Lower(int feature_id) const {
  return intervals_.find(feature_id)->second.lower.value();
}

bool BoundingBox::Contains(const Point& p) const {
  for (const auto& iter : intervals_) {
    if (!iter.second.Contains(p[iter.first]))
      return false;
  }
  return true;
}

Patch BoundingBox::ClosestPatchTo(const Point& point) const {
  Patch p;
  for (const auto& iter : Intervals()) {
    int feature_id = iter.first;
    const auto& inter = iter.second;
    p[feature_id] = inter.ClosestTo(point[feature_id]);
  }
  return std::move(p);
}

double BoundingBox::NormTo(const Point& point, int norm_type) const {
  auto patch = ClosestPatchTo(point);
  Point p(point);
  p.Apply(patch);
  return p.Norm(point, norm_type);
}

BoundingBox::IntervalsType& BoundingBox::Intervals() {
  return intervals_;
}

const BoundingBox::IntervalsType& BoundingBox::Intervals() const {
  return intervals_;
}

Interval& BoundingBox::operator[](int feature_id) {
  return intervals_[feature_id];
}

const Interval& BoundingBox::operator[](int feature_id) const {
  const auto& iter = intervals_.find(feature_id);
  assert(iter != intervals_.end());
  return iter->second;
}

Interval BoundingBox::GetOrEmpty(int feature_id) const {
  const auto& iter = intervals_.find(feature_id);
  if (iter == intervals_.end())
    return Interval();
  return iter->second;
}

bool BoundingBox::operator==(const BoundingBox& rhs) const {
  for (const auto& feature_interval : intervals_) {
    if (!(feature_interval.second == rhs[feature_interval.first]))
      return false;
  }
  return true;
}

const DecisionTree* BoundingBox::OwnerTree() const {
  return owner_tree_;
}

double BoundingBox::Label() const {
  return label_;
}

void BoundingBox::SetLabel(double label) {
  label_ = label;
}

std::string BoundingBox::ToDebugString() const {
  std::string str = "[";
  for (const auto& iter : intervals_)
    str += std::to_string(iter.first) + ":" + iter.second.ToDebugString() + ",";
  str += "]";
  return str;
}

void BoundingBox::Clear() {
  intervals_.clear();
  owner_tree_ = nullptr;
}

// LayeredBoundingBox

LayeredBoundingBox::LayeredBoundingBox(const DecisionForest* owner_forest,
                                       int num_class,
                                       int max_feature_id,
                                       int class1,
                                       int class2)
    : owner_forest_(owner_forest),
      ordered_boxes_(std::make_unique<OrderedBoxes>(max_feature_id)),
      scores_(num_class, 0),
      class1_(class1),
      class2_(class2) {}
LayeredBoundingBox::~LayeredBoundingBox() {}

void LayeredBoundingBox::AddBox(const BoundingBox* box) {
  assert(box->OwnerTree());
  assert(boxes_.find(box->OwnerTree()) == boxes_.end());

  boxes_[box->OwnerTree()] = box;
  hash_ ^= ptr_hasher_(box);
  scores_[box->OwnerTree()->ClassId()] += box->Label();

  ordered_boxes_->Add(box);
}

void LayeredBoundingBox::RemoveBox(const BoundingBox* box) {
  assert(box);
  const auto* owner_tree = box->OwnerTree();
  assert(owner_tree);
  boxes_.erase(owner_tree);
  hash_ ^= ptr_hasher_(box);
  scores_[owner_tree->ClassId()] -= box->Label();
}

int LayeredBoundingBox::PredictionLabel(
    const std::vector<double>* scores) const {
  if (!scores)
    scores = &scores_;

  if (class1_ == -1 && class2_ == -1)
    return MaxIndex(*scores);

  return MaxIndexBetween(*scores, class1_, class2_);
}

double LayeredBoundingBox::LabelScore(int victim_label,
                                      const std::vector<double>* scores) const {
  if (!scores)
    scores = &scores_;
  int adv_index = PredictionLabel(scores);
  if (adv_index == victim_label)
    return -10000;
  return (*scores)[adv_index] - (*scores)[victim_label];
}

const std::vector<double>& LayeredBoundingBox::Scores() const {
  return scores_;
}

std::vector<const BoundingBox*> LayeredBoundingBox::GetEffectiveBoxesForFeature(
    int feature_id,
    SearchMode search_mode) const {
  double upper_value = location_[feature_id];
  double lower_value = location_[feature_id];
  std::vector<const BoundingBox*> touching_boxes;

  // TODO: Only one of the branch should be executed, unless there is a
  // precision error.
  ordered_boxes_->FillIncompatibleBoxes(feature_id, upper_value + 2 * eps, true,
                                        &touching_boxes);
  ordered_boxes_->FillIncompatibleBoxes(feature_id, lower_value - 2 * eps,
                                        false, &touching_boxes);

  return std::move(touching_boxes);
}

std::vector<FeatureDir> LayeredBoundingBox::GetBoundedFeatures() const {
  int len = location_.Size();
  std::vector<FeatureDir> is_bounded(len, FeatureDir::None);

  const auto* cached_intersection = ordered_boxes_->GetCachedIntersection();
  for (int feature_id = 0; feature_id < len; ++feature_id) {
    if (cached_intersection->HasUpper(feature_id) &&
        location_[feature_id] + 2 * eps >=
            cached_intersection->Upper(feature_id)) {
      is_bounded[feature_id] = FeatureDir::Upper;
    } else if (cached_intersection->HasLower(feature_id) &&
               location_[feature_id] - 2 * eps <=
                   cached_intersection->Lower(feature_id)) {
      is_bounded[feature_id] = FeatureDir::Lower;
    }
  }

  return std::move(is_bounded);
}

std::vector<const BoundingBox*> LayeredBoundingBox::GetAlternativeBoxes(
    const BoundingBox& target_feature_constrain,
    int max_dist,
    const BoundingBox* box_to_replace,
    bool enable_relaxed_boundary,
    const BoundingBox* hard_constrain) const {
  std::vector<const BoundingBox*> alt_boxes;
  BoundingBox relaxed_box = target_feature_constrain;
  box_to_replace->OwnerTree()->CDfs(
      [&](const DecisionTree* t) -> std::pair<bool, bool> {
        if (t->is_leaf() && t->box() != box_to_replace) {
          alt_boxes.push_back(t->box());
          return {false, false};
        }

        auto iter = relaxed_box.Intervals().find(t->split_feature_id());
        if (iter == relaxed_box.Intervals().end()) {
          auto result = relaxed_box.Intervals().emplace(
              t->split_feature_id(),
              ordered_boxes_->GetKthInterval(
                  t->split_feature_id(), max_dist,
                  box_to_replace->GetOrEmpty(t->split_feature_id())));
          iter = result.first;
        }

        const auto& relaxed_bound = iter->second;

        const auto& left_bound = Interval::Upper(t->split_condition());
        bool go_left =
            relaxed_bound.Overlaps(left_bound) ||
            (enable_relaxed_boundary && relaxed_bound.Adjacents(left_bound));

        if (hard_constrain && go_left) {
          go_left =
              (*hard_constrain)[t->split_feature_id()].Overlaps(left_bound);
        }

        const auto& right_bound = Interval::Lower(t->split_condition());
        bool go_right =
            relaxed_bound.Overlaps(right_bound) ||
            (enable_relaxed_boundary && relaxed_bound.Adjacents(right_bound));

        if (hard_constrain && go_right) {
          go_right =
              (*hard_constrain)[t->split_feature_id()].Overlaps(right_bound);
        }

        return {go_left, go_right};
      });

  return std::move(alt_boxes);
}

void LayeredBoundingBox::FillIncompatibleBoxes(
    int feature_id,
    double value,
    std::vector<const BoundingBox*>* incompatible_boxes) const {
  if (value > location_[feature_id]) {
    ordered_boxes_->FillIncompatibleBoxes(feature_id, value, true,
                                          incompatible_boxes);
  } else if (value < location_[feature_id]) {
    ordered_boxes_->FillIncompatibleBoxes(feature_id, value, false,
                                          incompatible_boxes);
  }
}

Patch LayeredBoundingBox::StretchWithinBox(
    const Patch& patch,
    const Point& victim_point,
    const BoundingBox* constrain_box,
    const std::vector<const BoundingBox*>& incompatible_boxes) const {
  Timing::Instance()->BinCount("StretchWithinBox::constrain_box",
                               constrain_box != nullptr);

  Timing::Instance()->StartTimer(
      "LayeredBoundingBox::OptAdv::StretchWithinBox");

  Timing::Instance()->StartTimer(
      "LayeredBoundingBox::OptAdv::upper/lower_to_stretch::gen");

  std::set<int> upper_to_stretch;
  std::set<int> lower_to_stretch;
  // Handles |victim_point[feature_id] != location_[feature_id]|.
  for (const auto* box : incompatible_boxes) {
    for (const auto& iter : box->Intervals()) {
      int feature_id = iter.first;
      const auto& interval = iter.second;
      // |interval| may be looser than the bounding box.
      if (interval.upper.has_value() &&
          interval.upper.value() - location_[feature_id] <= 2 * eps) {
        upper_to_stretch.insert(feature_id);
      }
      if (interval.lower.has_value() &&
          location_[feature_id] - interval.lower.value() <= 2 * eps) {
        lower_to_stretch.insert(feature_id);
      }
    }
  }

  Timing::Instance()->EndTimer(
      "LayeredBoundingBox::OptAdv::upper/lower_to_stretch::gen");

  Patch new_adv_patch;

  Timing::Instance()->StartTimer("LayeredBoundingBox::OptAdv::stretch");

  for (int feature_id : upper_to_stretch) {
    if (victim_point[feature_id] <= location_[feature_id])
      continue;

    double new_bound =
        ordered_boxes_->GetUpperExcept(feature_id, incompatible_boxes)
            .value_or(DBL_MAX);

    double new_value = fmin(victim_point[feature_id], new_bound - eps);

    if (new_value > location_[feature_id])
      new_adv_patch[feature_id] = new_value;
  }

  for (int feature_id : lower_to_stretch) {
    if (victim_point[feature_id] >= location_[feature_id])
      continue;

    double new_bound =
        ordered_boxes_->GetLowerExcept(feature_id, incompatible_boxes)
            .value_or(-DBL_MAX);

    double new_value = fmax(victim_point[feature_id], new_bound + eps);

    if (new_value < location_[feature_id])
      new_adv_patch[feature_id] = new_value;
  }

  if (constrain_box) {
    for (const auto& iter : constrain_box->Intervals()) {
      int feature_id = iter.first;
      const auto& interval = iter.second;

      auto piter = new_adv_patch.find(feature_id);
      double current_value = (piter == new_adv_patch.end())
                                 ? location_[feature_id]
                                 : piter->second;

      if (!interval.Contains(current_value))
        new_adv_patch[feature_id] =
            interval.ClosestTo(victim_point[feature_id]);
    }
  }

  Timing::Instance()->EndTimer("LayeredBoundingBox::OptAdv::stretch");

  Timing::Instance()->BinCount("upper_to_stretch.size()",
                               upper_to_stretch.size());
  Timing::Instance()->BinCount("lower_to_stretch.size()",
                               lower_to_stretch.size());
  Timing::Instance()->BinCount("new_adv_patch.size()", new_adv_patch.size());

  Timing::Instance()->EndTimer("LayeredBoundingBox::OptAdv::StretchWithinBox");

  return std::move(new_adv_patch);
}

std::vector<const BoundingBox*> LayeredBoundingBox::GetNewBoxes(
    const Patch& patch,
    const std::vector<const BoundingBox*>& incompatible_boxes) const {
  Timing::Instance()->StartTimer("LayeredBoundingBox::GetNewBoxes");
  std::vector<const BoundingBox*> new_boxes;
  Point dummy_adv = location_;
  dummy_adv.Apply(patch);
  for (const auto* box : incompatible_boxes) {
    const BoundingBox* new_box = box->OwnerTree()->GetBoundingBox(dummy_adv);
    new_boxes.push_back(new_box);
  }
  Timing::Instance()->EndTimer("LayeredBoundingBox::GetNewBoxes");
  return std::move(new_boxes);
}

std::vector<double> LayeredBoundingBox::GetNewScores(
    const std::vector<const BoundingBox*>& incompatible_boxes,
    const std::vector<const BoundingBox*>& new_boxes) const {
  Timing::Instance()->StartTimer("LayeredBoundingBox::GetNewScores");
  std::vector<double> new_scores = scores_;
  for (int i = 0; i < incompatible_boxes.size(); ++i) {
    const auto* box = incompatible_boxes[i];
    const auto* new_box = new_boxes[i];
    new_scores[box->OwnerTree()->ClassId()] += new_box->Label() - box->Label();
  }
  Timing::Instance()->EndTimer("LayeredBoundingBox::GetNewScores");
  return std::move(new_scores);
}

void LayeredBoundingBox::TightenPoint(
    Point* new_adv,
    const std::vector<const BoundingBox*>& new_boxes) const {
  Timing::Instance()->StartTimer("LayeredBoundingBox::OptAdv::TightenPoint");

  for (const auto* new_box : new_boxes) {
    for (const auto& iter : new_box->Intervals()) {
      int feature_id = iter.first;
      const auto& interval = iter.second;
      if (interval.upper.has_value()) {
        (*new_adv)[feature_id] =
            fmin((*new_adv)[feature_id], interval.upper.value() - eps);
      }

      if (interval.lower.has_value()) {
        (*new_adv)[feature_id] =
            fmax((*new_adv)[feature_id], interval.lower.value() + eps);
      }
    }
  }

  Timing::Instance()->EndTimer("LayeredBoundingBox::OptAdv::TightenPoint");
}

void LayeredBoundingBox::ShiftPoint(const Point& point) {
  ShiftByPatch(point.Diff(location_));
}

void LayeredBoundingBox::ShiftByPatch(const Patch& patch) {
  Timing::Instance()->StartTimer("LayeredBoundingBox::ShiftByPatch");

  Timing::Instance()->StartTimer("LayeredBoundingBox::removed_boxes");
  std::vector<const BoundingBox*> removed_boxes;

  for (const auto& feature_value : patch) {
    int feature_id = feature_value.first;
    double value = feature_value.second;

    if (value > location_[feature_id]) {
      Timing::Instance()->StartTimer("LayeredBoundingBox::RemoveUpperUntil");
      auto tmp = ordered_boxes_->RemoveUntil(feature_id, value, true);
      Timing::Instance()->EndTimer("LayeredBoundingBox::RemoveUpperUntil");
      removed_boxes.insert(removed_boxes.end(), tmp.begin(), tmp.end());
    } else if (value < location_[feature_id]) {
      Timing::Instance()->StartTimer("LayeredBoundingBox::RemoveLowerUntil");
      auto tmp = ordered_boxes_->RemoveUntil(feature_id, value, false);
      Timing::Instance()->EndTimer("LayeredBoundingBox::RemoveLowerUntil");
      removed_boxes.insert(removed_boxes.end(), tmp.begin(), tmp.end());
    }
  }
  location_.Apply(patch);
  Timing::Instance()->EndTimer("LayeredBoundingBox::removed_boxes");

  Timing::Instance()->BinCount("NumRemovedBoxes", removed_boxes.size());

  Timing::Instance()->StartTimer("LayeredBoundingBox::RemoveBox");
  for (const auto* box : removed_boxes) {
    RemoveBox(box);
  }
  Timing::Instance()->EndTimer("LayeredBoundingBox::RemoveBox");

  Timing::Instance()->StartTimer("LayeredBoundingBox::AddBox");
  for (const auto* box : removed_boxes) {
    AddBox(box->OwnerTree()->GetBoundingBox(location_));
  }
  Timing::Instance()->EndTimer("LayeredBoundingBox::AddBox");

  // assert(cached_intersection_.Contains(new_point));

  Timing::Instance()->EndTimer("LayeredBoundingBox::ShiftByPatch");
}

const BoundingBox* LayeredBoundingBox::GetCachedIntersection() const {
  return ordered_boxes_->GetCachedIntersection();
}

std::vector<BoundingBox> LayeredBoundingBox::GetIndenpendentBoundingBoxes()
    const {
  std::vector<BoundingBox> re;

  for (const auto& iter : boxes_) {
    const auto* box = iter.second;
    BoundingBox tmp(box->OwnerTree());
    for (const auto& iter : box->Intervals()) {
      const auto& intersection = (*GetCachedIntersection())[iter.first];
      const auto& current_inter = iter.second;
      if (current_inter.HasStricterUpper(intersection) &&
          current_inter.HasStricterLower(intersection)) {
        tmp.IntersectFeature(iter.first, current_inter);
      } else if (iter.second.HasStricterUpper(intersection)) {
        tmp.IntersectFeature(iter.first,
                             Interval::Upper(current_inter.upper.value()));
      } else if (iter.second.HasStricterLower(intersection)) {
        tmp.IntersectFeature(iter.first,
                             Interval::Lower(current_inter.lower.value()));
      }
    }

    if (tmp.Intervals().size() > 0)
      re.emplace_back(std::move(tmp));
  }

  return std::move(re);
}

void LayeredBoundingBox::ShiftByDirection(const Direction& dir) {
  Patch p;
  for (int d : dir) {
    int feature_id = abs(d);
    const auto& feature_interval =
        GetCachedIntersection()->Intervals().find(feature_id);
    if (feature_interval == GetCachedIntersection()->Intervals().end())
      return;

    const auto& interval = feature_interval->second;
    if (d > 0 && interval.upper.has_value()) {
      p[feature_id] = interval.upper.value() + eps;
    } else if (d < 0 && interval.lower.has_value()) {
      p[feature_id] = interval.lower.value() - eps;
    }
  }
  ShiftByPatch(std::move(p));
}

void LayeredBoundingBox::SetInitialLocation(const Point& initial_location) {
  assert(location_.Size() == 0);
  location_ = initial_location;
}

const Point& LayeredBoundingBox::Location() const {
  return location_;
}

void LayeredBoundingBox::VerifyCachedIntersectionForTesting() const {
  BoundingBox intersection;

  for (const auto& tree_box : boxes_) {
    intersection.Intersect(*(tree_box.second));
  }

  assert(intersection == *GetCachedIntersection());
}

size_t LayeredBoundingBox::Hash() const {
  return hash_;
}

const BoundingBox* LayeredBoundingBox::GetBoxForTree(
    const DecisionTree* tree) const {
  return boxes_.find(tree)->second;
}

std::vector<const BoundingBox*> LayeredBoundingBox::GetBoxForAllTree() const {
  std::vector<const BoundingBox*> all_box;
  all_box.reserve(boxes_.size());
  for (const auto& iter : boxes_) {
    all_box.push_back(iter.second);
  }
  return std::move(all_box);
}

bool LayeredBoundingBox::CheckScoresForTesting(
    const Patch& patch,
    const std::vector<double>& scores) const {
  Point tmp_adv(location_);
  tmp_adv.Apply(patch);
  auto correct_scores = owner_forest_->ComputeScores(tmp_adv);
  for (int i = 0; i < correct_scores.size(); ++i) {
    assert(fabs(correct_scores[i] - scores[i]) < 1e-5);
  }
  return true;
}

void LayeredBoundingBox::AssertTightForTesting(
    const Point& victim_point) const {
  for (int feature_id = 0; feature_id < victim_point.Size(); ++feature_id) {
    if (victim_point[feature_id] > location_[feature_id]) {
      assert(GetCachedIntersection()->HasUpper(feature_id));
      assert(GetCachedIntersection()->Upper(feature_id) -
                 location_[feature_id] <=
             2 * eps);
    } else if (victim_point[feature_id] < location_[feature_id]) {
      assert(GetCachedIntersection()->HasLower(feature_id));
      assert(location_[feature_id] -
                 GetCachedIntersection()->Lower(feature_id) <=
             2 * eps);
    }
  }
}

}  // namespace cz
