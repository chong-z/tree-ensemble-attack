#include "neighbor_attack.h"

#include <algorithm>
#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <random>

#include <boost/lexical_cast.hpp>

#include "bounding_box.h"
#include "decision_forest.h"
#include "decision_tree.h"
#include "interval.h"
#include "timing.h"

using std::cout;
using std::endl;

namespace cz {

namespace {

class DualNormScore : public cz::NeighborScore {
 public:
  DualNormScore(int norm_type_1)
      : norm_type_1_(norm_type_1), norm_type_2_(norm_type_1 == 2 ? -1 : 2) {}

  ScoreType Slow(const Point& point,
                 const Point& ref_point) const override {
    Timing::Instance()->StartTimer("DualNormScore::Slow");
    double norm1 = point.Norm(ref_point, norm_type_1_);
    double norm2 = point.Norm(ref_point, norm_type_2_);
    Timing::Instance()->EndTimer("DualNormScore::Slow");
    return {-norm1, -norm2};
  }

  ScoreType Fast(const ScoreType& old_score,
                 const Point& old_point,
                 const Point& ref_point,
                 const Patch& new_patch) const override {
    Timing::Instance()->StartTimer("DualNormScore::Fast");
    double norm1 = NormFast(-old_score.first, old_point, ref_point, new_patch,
                            norm_type_1_);
    double norm2 = NormFast(-old_score.second, old_point, ref_point, new_patch,
                            norm_type_2_);
    Timing::Instance()->EndTimer("DualNormScore::Fast");
    return {-norm1, -norm2};
  }

 private:
  int norm_type_1_;
  int norm_type_2_;
};

template <class CIter>
std::list<Patch> GeneratePatches(CIter& iter,
                                 const CIter& end,
                                 const std::vector<FeatureDir>& is_bounded,
                                 const BoundingBox* cached_intersection) {
  FeatureDir effective_dir = FeatureDir::None;
  while (iter != end) {
    if ((is_bounded[iter->first] & FeatureDir::Upper) &&
        iter->second.upper.has_value()) {
      if (!cached_intersection ||
          iter->second.HasSameUpper((*cached_intersection)[iter->first]))
        effective_dir |= FeatureDir::Upper;
    }

    if ((is_bounded[iter->first] & FeatureDir::Lower) &&
        iter->second.lower.has_value()) {
      if (!cached_intersection ||
          iter->second.HasSameLower((*cached_intersection)[iter->first]))
        effective_dir |= FeatureDir::Lower;
    }

    if (effective_dir != FeatureDir::None)
      break;

    ++iter;
  }

  if (iter == end)
    return {Patch()};

  int feature_id = iter->first;
  const Interval& inter = iter->second;
  std::vector<double> new_values;
  if (effective_dir & FeatureDir::Upper) {
    if (cached_intersection) {
      new_values.push_back(cached_intersection->Upper(feature_id) + eps);
    } else {
      new_values.push_back(inter.upper.value() + eps);
    }
  }
  if (effective_dir & FeatureDir::Lower) {
    if (cached_intersection) {
      new_values.push_back(cached_intersection->Lower(feature_id) - eps);
    } else {
      new_values.push_back(inter.lower.value() - eps);
    }
  }

  ++iter;
  std::list<Patch> patches =
      GeneratePatches(iter, end, is_bounded, cached_intersection);
  std::list<Patch> new_patches;

  for (double new_value : new_values) {
    for (const auto& p : patches) {
      Patch tmp(p);
      tmp[feature_id] = new_value;
      new_patches.emplace_back(std::move(tmp));
    }
  }
  new_patches.splice(new_patches.end(), patches);

  return std::move(new_patches);
}

void GenerateNaivePatches(SearchMode search_mode,
                          const Point& starting_point,
                          const BoundingBox* current_box,
                          const std::vector<FeatureDir>& is_bounded,
                          const BoundingBox* cached_intersection,
                          std::vector<Patch>* patches) {
  if (search_mode == SearchMode::NaiveLeaf) {
    auto leaves = current_box->OwnerTree()->GetLeaves();
    for (const auto* leaf : leaves) {
      patches->emplace_back(leaf->ClosestPatchTo(starting_point));
    }
    return;
  }

  assert(search_mode == SearchMode::NaiveFeature);
  for (const auto& iter : current_box->Intervals()) {
    int feature_id = iter.first;
    const auto& inter = iter.second;

    if ((is_bounded[feature_id] & FeatureDir::Upper) &&
        inter.HasSameUpper((*cached_intersection)[feature_id])) {
      patches->push_back({{feature_id, inter.upper.value() + eps}});
    } else if ((is_bounded[feature_id] & FeatureDir::Lower) &&
               inter.HasSameLower((*cached_intersection)[feature_id])) {
      patches->push_back({{feature_id, inter.lower.value() - eps}});
    }
  }
}

Patch GenRandomDir(int feature_size) {
  Patch dir;
  const int kNumDir = 5;
  while (dir.size() < kNumDir) {
    int feature_id = rand() % feature_size;
    int feature_dir = (rand() % 2) * 2 - 1;
    if (dir.find(feature_id) == dir.end())
      dir[feature_id] = feature_dir;
  }

  return std::move(dir);
}

Patch ComputeRecoveryPatch(const Point& point, const BoundingBox& box) {
  Patch p;
  for (const auto& iter : box.Intervals()) {
    p[iter.first] = point[iter.first];
  }
  return std::move(p);
}

Patch ComputeRecoveryPatch(const Point& point, const Direction& dir) {
  Patch p;
  for (auto d : dir) {
    p[abs(d)] = point[abs(d)];
  }
  return std::move(p);
}

bool HasSameLabel(int l1, int l2) {
  return l1 == l2;
}

void OptimizeLinearSearch(LayeredBoundingBox* layered_box,
                          const Point& victim_point,
                          int victim_label) {
  while (true) {
    Point adv_point = victim_point;

    const auto* box = layered_box->GetCachedIntersection();
    for (const auto& iter : box->Intervals()) {
      int feature_id = iter.first;
      if (iter.second.upper.has_value()) {
        adv_point[feature_id] =
            fmin(adv_point[feature_id], iter.second.upper.value() + eps);
      }

      if (iter.second.lower.has_value()) {
        adv_point[feature_id] =
            fmax(adv_point[feature_id], iter.second.lower.value() - eps);
      }
    }

    Point original_point = layered_box->Location();
    layered_box->ShiftPoint(adv_point);
    if (HasSameLabel(layered_box->PredictionLabel(), victim_label)) {
      layered_box->ShiftPoint(original_point);
      return;
    }
  }
}

std::pair<double, double> NormFirstScore(
    int victim_label,
    double norm,
    const std::vector<double>& label_scores) {
  int adv_index = MaxIndex(label_scores, victim_label);
  return std::make_pair(-norm,
                        label_scores[adv_index] - label_scores[victim_label]);
}

std::pair<double, double> LabelFirstScore(
    int victim_label,
    double norm,
    const std::vector<double>& label_scores) {
  int adv_index = MaxIndex(label_scores, victim_label);
  return std::make_pair(label_scores[adv_index] - label_scores[victim_label],
                        -norm);
}

std::pair<double, double> WeightedScore(
    int victim_label,
    double norm_weight,
    double norm,
    const std::vector<double>& label_scores) {
  int adv_index = MaxIndex(label_scores, victim_label);
  double label_score = label_scores[adv_index] - label_scores[victim_label];
  double score = norm_weight * label_score + label_score / (norm + 2.0);
  // double score = norm_weight * (-norm) + (1 - norm_weight) * label_score;
  return std::make_pair(score, 0);
}

bool True() {
  return true;
}

std::vector<const BoundingBox*> GetIncompatibleBoxes(
    const Patch& patch,
    const LayeredBoundingBox* layered_box,
    std::map<std::pair<int, double>, std::vector<const BoundingBox*>>*
        cached_incompatible_boxes_per_feature) {
  Timing::Instance()->StartTimer(
      "GetIncompatibleBoxes::cached_incompatible_boxes_per_feature");
  std::vector<const BoundingBox*> incompatible_boxes;
  for (const auto& feature_value : patch) {
    auto iter = cached_incompatible_boxes_per_feature->find(feature_value);
    if (iter != cached_incompatible_boxes_per_feature->end()) {
      incompatible_boxes.insert(incompatible_boxes.end(), iter->second.begin(),
                                iter->second.end());
    } else {
      std::vector<const BoundingBox*> boxes;
      layered_box->FillIncompatibleBoxes(feature_value.first,
                                         feature_value.second, &boxes);
      incompatible_boxes.insert(incompatible_boxes.end(), boxes.begin(),
                                boxes.end());
      cached_incompatible_boxes_per_feature->emplace(feature_value,
                                                     std::move(boxes));
    }
  }
  Timing::Instance()->EndTimer(
      "GetIncompatibleBoxes::cached_incompatible_boxes_per_feature");

  Timing::Instance()->StartTimer("GetIncompatibleBoxes::sort");

  std::sort(incompatible_boxes.begin(), incompatible_boxes.end());
  auto last = std::unique(incompatible_boxes.begin(), incompatible_boxes.end());
  incompatible_boxes.erase(last, incompatible_boxes.end());
  Timing::Instance()->EndTimer("GetIncompatibleBoxes::sort");

  Timing::Instance()->BinCount("incompatible_boxes.size()",
                               incompatible_boxes.size());

  return std::move(incompatible_boxes);
}

}  // namespace

const std::vector<int> NeighborAttack::kAllowedNormTypes = {-1, 1, 2};

NeighborAttack::NeighborAttack(const Config& config) : config_(config) {}

NeighborAttack::~NeighborAttack() {}

void NeighborAttack::LoadForestFromJson(const std::string& path) {
  assert(!forest_);
  forest_ = DecisionForest::CreateFromJson(
      path, config_.num_classes,
      config_.num_features + config_.feature_start - 1);

  // Also load train data if it's RBA-Appr (Yang et al. 2019).
  if (config_.search_mode == SearchMode::Region) {
    auto train_list =
        cz::LoadSVMFile(config_.train_path.c_str(), config_.num_features,
                        config_.feature_start);
    // Move to vector for better parallelize.
    train_data_.insert(train_data_.end(), train_list.begin(), train_list.end());
  }
}

NeighborAttack::Result NeighborAttack::FindAdversarialPoint(
    const Point& victim_point) const {

  int victim_label = forest_->PredictLabel(victim_point);
  printf("NeighborAttack::FindAdversarialPoint Spawning %d threads\n",
         config_.num_threads);
  boost::asio::thread_pool pool(config_.num_threads);

  int num_work = config_.num_attack_per_point;
  if (config_.search_mode == SearchMode::Region)
    num_work = config_.num_threads;

  Result results[num_work];
  for (int i = 0; i < num_work; ++i) {
    boost::asio::post(
        pool, boost::bind(&NeighborAttack::FindAdversarialPoint_ThreadRun, this,
                          i, victim_point, victim_label, &results[i]));
  }
  pool.join();

  Result best_result;
  for (int i = 0; i < num_work; ++i) {
    if (!results[i].success())
      continue;

    best_result.hist_points.push_back(
        results[i].best_points[config_.norm_type]);
    for (const int norm_type : kAllowedNormTypes) {
      double new_norm = results[i].best_norms[norm_type];
      if (new_norm < best_result.best_norms[norm_type]) {
        best_result.best_norms[norm_type] = new_norm;
        best_result.best_points[norm_type] = results[i].best_points[norm_type];
      }
    }
  }

  return std::move(best_result);
}

void NeighborAttack::FindAdversarialPoint_ThreadRun(int task_id,
                                                    const Point& victim_point,
                                                    int victim_label,
                                                    Result* result) const {
  printf(
      "NeighborAttack::FindAdversarialPoint_ThreadRun Trying %d/%d random "
      "starting points... thread_id:%s\n",
      task_id + 1, config_.num_attack_per_point,
      boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str());

  if (config_.search_mode == SearchMode::Region) {
    return RegionBasedAttackAppr_ThreadRun(task_id, victim_point, victim_label,
                                           result);
  }

  // TODO: Improve the quality of |GenFromVictim| points.
  bool use_random = true;
  Point p;

  Timing::Instance()->StartTimer("GenInitialPoint");
  if (use_random) {
    auto res = NormalRandomPoint(victim_label, victim_point, task_id);
    if (res.first) {
      p = res.second;
    } else {
      Timing::Instance()->EndTimer("GenInitialPoint");
      printf("  Failed to generate %d/%d random starting points... thread_id:%s\n",
        task_id + 1, config_.num_attack_per_point,
        boost::lexical_cast<std::string>(boost::this_thread::get_id()).c_str());
      return;
    }
    // p = FeatureSplitsRandomPoint(victim_point, victim_label, task_id);
    // p = PureRandomPoint(victim_point, victim_label);
  } else {
    assert(false);
    // double norm_weight =
    //     task_id * 1.0 / fmax(1, config_.num_attack_per_point - 1);
    // cout << "norm_weight:" << norm_weight << endl;
    // auto opt_p = GenFromVictim(victim_point, victim_label, norm_weight);
    // if (opt_p.has_value()) {
    //   p = std::move(opt_p.value());
    // } else {
    //   cout << "Fall back to regular random points..." << endl;
    //   p = PureRandomPoint(victim_point, victim_label, task_id);
    // }
  }
  Timing::Instance()->EndTimer("GenInitialPoint");

  int target_label = forest_->PredictLabel(p);

  printf("Initial point label:%d\n",
         forest_->PredictLabelBetween(p, target_label, victim_label));
  DCHECK(
      !HasSameLabel(forest_->PredictLabelBetween(p, target_label, victim_label),
                    victim_label));

  p = OptimizeAdversarialPoint(p, victim_point, target_label, victim_label,
                               result);
  DCHECK(
      !HasSameLabel(forest_->PredictLabelBetween(p, target_label, victim_label),
                    victim_label));

  printf("Norms for random point %d:%s\n", task_id + 1,
         GetNormStringForLogging(p, victim_point).c_str());

  UpdateResult(p, victim_point, result);
}

int NeighborAttack::PredictLabel(const Point& point) const {
  return forest_->PredictLabel(point);
}

int NeighborAttack::HammingDistanceBetween(const Point& p1, const Point& p2, int class1, int class2) const {
  return forest_->HammingDistanceBetween(p1, p2, class1, class2);
}

int NeighborAttack::NeighborDistanceBetween(const Point& start_point,
                                            const Point& end_point,
                                            int adv_class,
                                            int victim_class,
                                            const Point& victim_point) const {
  int num_trials = 200;

  boost::asio::thread_pool pool(config_.num_threads);
  int results[num_trials];
  for (int i = 0; i < num_trials; ++i) {
    boost::asio::post(
        pool, boost::bind(&NeighborAttack::NeighborDistanceBetween_ThreadRun,
                          this, i, start_point, end_point, adv_class,
                          victim_class, victim_point, &results[i]));
  }
  pool.join();

  int neighbor_dist = *std::min_element(results, results + num_trials);

  return neighbor_dist;
}

void NeighborAttack::NeighborDistanceBetween_ThreadRun(
    int task_id,
    const Point& start_point,
    const Point& end_point,
    int adv_class,
    int victim_class,
    const Point& victim_point,
    int* out_neighbor_dist) const {
  int norm_type1 = config_.norm_type;
  int norm_type2 = norm_type1 == 2 ? -1 : 2;

  auto start_scores = forest_->ComputeScores(start_point);
  auto end_scores = forest_->ComputeScores(end_point);
  // End point must be valid and better adv points.
  assert(end_scores[adv_class] > end_scores[victim_class]);

  ScoreType current_norm{victim_point.Norm(start_point, norm_type1),
                         victim_point.Norm(start_point, norm_type2)};

  auto end_box =
      forest_->GetLayeredBoundingBox(end_point, adv_class, victim_class);
  ScoreType end_norm{
      end_box->GetCachedIntersection()->NormTo(victim_point, norm_type1),
      end_box->GetCachedIntersection()->NormTo(victim_point, norm_type2)};

  if (current_norm <= end_norm) {
    // The |start_point| is already better.
    *out_neighbor_dist = 0;
    return;
  }

  // assert(victim_point.Norm(start_point, norm_type) >
  // victim_point.Norm(end_point, norm_type));

  // Positive: victim_class; Negative: adv_class.
  // |initial_label| may be negative in multi-class case.
  double initial_label = start_scores[victim_class] - start_scores[adv_class];

  // Adv boxes goes first.
  using TupleType = std::tuple<double, const BoundingBox*, const BoundingBox*>;
  std::vector<TupleType> negative_tuples;
  std::vector<TupleType> positive_tuples;

  BoundingBox shared_box;

  for (const auto& t : forest_->trees_) {
    if (t->ClassId() != adv_class && t->ClassId() != victim_class)
      continue;

    const auto* sb = t->GetBoundingBox(start_point);
    const auto* eb = t->GetBoundingBox(end_point);

    if (sb != eb) {
      double label_diff = eb->Label() - sb->Label();
      if (t->ClassId() == adv_class)
        label_diff = -label_diff;
      if (label_diff < 0) {
        negative_tuples.push_back({label_diff, sb, eb});
      } else {
        positive_tuples.push_back({label_diff, sb, eb});
      }
    } else {
      shared_box.Intersect(*sb);
    }
  }

  double current_label = initial_label;

  if (task_id == 0) {
    std::sort(negative_tuples.begin(), negative_tuples.end());
    std::sort(positive_tuples.begin(), positive_tuples.end());
  } else {
    auto rng = std::default_random_engine(task_id);
    std::shuffle(negative_tuples.begin(), negative_tuples.end(), rng);
    std::shuffle(positive_tuples.begin(), positive_tuples.end(), rng);
  }

  std::list<TupleType> changed_list;
  std::list<TupleType> remaining_list;
  remaining_list.insert(remaining_list.end(), negative_tuples.begin(),
                        negative_tuples.end());
  remaining_list.insert(remaining_list.end(), positive_tuples.begin(),
                        positive_tuples.end());

  int max_neighbor_dist = 0;
  while (!remaining_list.empty()) {
    int current_neighbor_dist = 0;
    bool should_add_first_node = true;
    bool added_new_node = false;
    BoundingBox current_box;

    while (should_add_first_node || current_label > 0 || added_new_node) {
      added_new_node = false;
      if (current_neighbor_dist == 0 || current_label > 0)
        assert(!remaining_list.empty());

      if (should_add_first_node || current_label > 0) {
        should_add_first_node = false;
        auto node = remaining_list.front();
        remaining_list.pop_front();
        assert(current_label < 0 || std::get<0>(node) < 0);
        current_label += std::get<0>(node);
        current_box.Intersect(*std::get<2>(node));
        added_new_node = true;
        ++current_neighbor_dist;
        changed_list.emplace_back(std::move(node));
      }

      auto iter = remaining_list.begin();
      while (iter != remaining_list.end()) {
        if (!current_box.Overlaps(*std::get<1>(*iter))) {
          auto node = *iter;
          iter = remaining_list.erase(iter);
          current_label += std::get<0>(node);
          assert(current_box.Overlaps(*std::get<2>(node)));
          current_box.Intersect(*std::get<2>(node));
          added_new_node = true;
          ++current_neighbor_dist;
          changed_list.emplace_back(std::move(node));
        } else {
          ++iter;
        }
      }

      if (!added_new_node) {
        auto merged_box = shared_box;
        for (const auto& node : remaining_list)
          merged_box.Intersect(*std::get<1>(node));
        for (const auto& node : changed_list)
          merged_box.Intersect(*std::get<2>(node));

        ScoreType new_norm{merged_box.NormTo(victim_point, norm_type1),
                           merged_box.NormTo(victim_point, norm_type2)};
        if ((new_norm >= current_norm || new_norm <= end_norm) &&
            !remaining_list.empty()) {
          should_add_first_node = true;
        }
      }
    }

    auto merged_box = shared_box;
    for (const auto& node : remaining_list)
      merged_box.Intersect(*std::get<1>(node));
    for (const auto& node : changed_list)
      merged_box.Intersect(*std::get<2>(node));
    ScoreType new_norm{merged_box.NormTo(victim_point, norm_type1),
                       merged_box.NormTo(victim_point, norm_type2)};
    assert(new_norm < current_norm && new_norm >= end_norm);
    current_norm = new_norm;

    max_neighbor_dist = std::max(max_neighbor_dist, current_neighbor_dist);
  }

  *out_neighbor_dist = max_neighbor_dist;
}

void NeighborAttack::RegionBasedAttackAppr_ThreadRun(int task_id,
                                                     const Point& victim_point,
                                                     int victim_label,
                                                     Result* result) const {
  int step = config_.num_threads;
  for (int i = task_id; i < train_data_.size(); i += step) {
    auto train = train_data_[i];
    // int y_train = train.first;
    const auto& x_train = train.second;
    // |y_pred| may be different from |y_train|.
    int y_pred = PredictLabel(x_train);
    if (y_pred == victim_label)
      continue;
    auto joint_box = forest_->GetBoundingBox(x_train);
    auto optimze_adv = OptimizeLocalSearch(&joint_box, victim_point);

    UpdateResult(optimze_adv, victim_point, result);
  }
}

const DecisionForest* NeighborAttack::ForestForTesting() const {
  return forest_.get();
}

Point NeighborAttack::OptimizeAdversarialPoint(Point adv_point,
                                               const Point& victim_point,
                                               int target_label,
                                               int victim_label,
                                               Result* result) const {
  adv_point =
      OptimizeBinarySearch(adv_point, victim_point, target_label, victim_label);

  auto layered_box =
      forest_->GetLayeredBoundingBox(adv_point, target_label, victim_label);
  // TODO: Could remove?
  adv_point =
      OptimizeLocalSearch(layered_box->GetCachedIntersection(), victim_point);
  layered_box->ShiftPoint(adv_point);

  // Timing::Instance()->StartTimer("OptimizeLinearSearch1");
  // OptimizeLinearSearch(layered_box.get(), victim_point, victim_label);
  // adv_point = layered_box->Location();
  // Timing::Instance()->EndTimer("OptimizeLinearSearch1");

  bool found_better_adv = true;
  auto best_norm = Norm(adv_point, victim_point);
  int i = 0;

  const int kMaxRandomFallback = 20;
  int used_random_fallback = 0;

  while (found_better_adv) {
    found_better_adv = false;
    if (config_.collect_histogram) {
      printf(
          "NeighborAttack::OptimizeAdversarialPoint iteration: %d best_norm: "
          "%lf\n   %s\n",
          i++, best_norm,
          GetNormStringForLogging(adv_point, victim_point).c_str());
    }

    Timing::Instance()->StartTimer("OptimizeNeighborSearch");
    found_better_adv =
        OptimizeNeighborSearch(layered_box.get(), victim_point, target_label,
                               victim_label, DualNormScore(config_.norm_type),
                               config_.search_mode, config_.max_dist);
    Timing::Instance()->EndTimer("OptimizeNeighborSearch");

    adv_point = layered_box->Location();
    DCHECK(!HasSameLabel(
        forest_->PredictLabelBetween(adv_point, target_label, victim_label),
        victim_label));

    // Add randomness to jump out of local minimum.
    if (!found_better_adv && used_random_fallback < kMaxRandomFallback) {
      ++used_random_fallback;
      auto res = OptimizeRandomSearch(adv_point, victim_point, victim_label);
      found_better_adv = res.first;

      if (found_better_adv) {
        target_label = forest_->PredictLabel(res.second);
        layered_box = forest_->GetLayeredBoundingBox(res.second, target_label,
                                                     victim_label);
        adv_point = OptimizeLocalSearch(layered_box->GetCachedIntersection(),
                                        victim_point);
        layered_box->ShiftPoint(adv_point);
      }
    }

    UpdateResult(adv_point, victim_point, result);

    auto new_norm = Norm(adv_point, victim_point);

    if (found_better_adv) {
      // |new_norm| could be larger than |best_norm| on non-inf norms due to
      // |OptimizeRandomSearch|.
      // assert(config_.norm_type != -1 || new_norm <= best_norm);
      best_norm = fmin(best_norm, new_norm);
    }
  }

  // printf("OptimizeAdversarialPoint::used_random_fallback:%d\n",
         // used_random_fallback);
  Timing::Instance()->BinCount("OptimizeAdversarialPoint::used_random_fallback",
                               used_random_fallback);
  Timing::Instance()->BinCount("OptimizeAdversarialPoint::Iterations", i);

  return std::move(adv_point);
}

Point NeighborAttack::OptimizeLocalSearch(const BoundingBox* box,
                                          const Point& victim_point) const {
  Timing::Instance()->StartTimer("NeighborAttack::OptimizeLocalSearch");

  Point adv_point = victim_point;

  for (const auto& iter : box->Intervals()) {
    int feature_id = iter.first;
    if (iter.second.upper.has_value()) {
      adv_point[feature_id] =
          fmin(adv_point[feature_id], iter.second.upper.value() - eps);
    }

    if (iter.second.lower.has_value()) {
      adv_point[feature_id] =
          fmax(adv_point[feature_id], iter.second.lower.value() + eps);
    }
  }

  Timing::Instance()->EndTimer("NeighborAttack::OptimizeLocalSearch");
  return std::move(adv_point);
}

Point NeighborAttack::OptimizeBinarySearch(const Point& adv_point,
                                           const Point& victim_point,
                                           int target_label,
                                           int victim_label) const {
  Timing::Instance()->StartTimer("NeighborAttack::OptimizeBinarySearch");
  if (config_.collect_histogram) {
    printf("Starting binary search, %s\n",
           GetNormStringForLogging(adv_point, victim_point).c_str());
  }

  int iterations = 0;

  Point small_diff = (adv_point - victim_point) / 100.0;
  Point upper = victim_point;
  while (forest_->PredictLabelBetween(upper, target_label, victim_label) ==
         victim_label) {
    ++iterations;
    upper += small_diff;
  }

  if (config_.collect_histogram) {
    printf("Finished linear search, iterations=%d %s\n", iterations,
           GetNormStringForLogging(adv_point, victim_point).c_str());
  }

  Point lower = victim_point;

  while ((upper - lower).Norm(-1) > config_.binary_search_threshold) {
    ++iterations;
    Point mid = (upper + lower) / 2;
    if (forest_->PredictLabelBetween(mid, target_label, victim_label) ==
        victim_label) {
      lower = std::move(mid);
    } else {
      upper = std::move(mid);
    }
  }

  if (config_.collect_histogram) {
    printf("Finished binary search, iterations=%d %s\n", iterations,
           GetNormStringForLogging(adv_point, victim_point).c_str());
  }
  Timing::Instance()->BinCount("BinarySearchIterations", iterations);

  Timing::Instance()->EndTimer("NeighborAttack::OptimizeBinarySearch");

  return upper;
}

bool NeighborAttack::OptimizeNeighborSearch(LayeredBoundingBox* layered_box,
                                            const Point& victim_point,
                                            int target_label,
                                            int victim_label,
                                            const NeighborScore& neighbor_score,
                                            SearchMode search_mode,
                                            int max_dist) const {
  Timing::Instance()->StartTimer("OptimizeNeighborSearch::Setup");
  const Point& starting_point = layered_box->Location();
  const auto& starting_score =
      neighbor_score.Slow(starting_point, victim_point);

  bool found_better_adv = false;
  Point best_adv;
  auto best_score = starting_score;
  L(Patch best_patch);
  Timing::Instance()->EndTimer("OptimizeNeighborSearch::Setup");

  Timing::Instance()->StartTimer("OptimizeNeighborSearch::bounded_features");
  const int kFeatureSize = victim_point.Size();
  std::vector<FeatureDir> is_bounded(kFeatureSize, FeatureDir::None);
  // <distance, feature_id>
  std::vector<std::pair<double, int>> bounded_features;

  // Compute |is_bounded| and |bounded_features|.
  for (const auto& iter : layered_box->GetCachedIntersection()->Intervals()) {
    int feature_id = iter.first;
    if (victim_point[feature_id] > starting_point[feature_id]) {
      is_bounded[feature_id] = FeatureDir::Upper;
      bounded_features.push_back(std::make_pair(
          std::abs(starting_point[feature_id] - victim_point[feature_id]),
          feature_id));
    } else if (victim_point[feature_id] < starting_point[feature_id]) {
      is_bounded[feature_id] = FeatureDir::Lower;
      bounded_features.push_back(std::make_pair(
          std::abs(starting_point[feature_id] - victim_point[feature_id]),
          feature_id));
    }
  }

  Timing::Instance()->BinCount("bounded_features.size()",
                               bounded_features.size());
  Timing::Instance()->EndTimer("OptimizeNeighborSearch::bounded_features");

  std::set<const BoundingBox*> visited_box;
  std::set<Patch, PatchCompare> visited_patch;

  int early_return_interval;
  int current_iteration = 0;
  if (config_.enable_early_return) {
    Timing::Instance()->StartTimer(
        "OptimizeNeighborSearch::kEnableEarlyReturn");
    early_return_interval = 1;
    std::sort(bounded_features.begin(), bounded_features.end(),
              std::greater<>());
    Timing::Instance()->EndTimer("OptimizeNeighborSearch::kEnableEarlyReturn");
  }

  std::map<std::pair<int, double>, std::vector<const BoundingBox*>>
      cached_incompatible_boxes_per_feature;

  for (const auto& dist_feature : bounded_features) {
    const int feature_id = dist_feature.second;

    Timing::Instance()->StartTimer(
        "OptimizeNeighborSearch::GetEffectiveBoxesForFeature");
    auto effective_boxes =
        layered_box->GetEffectiveBoxesForFeature(feature_id, search_mode);
    Timing::Instance()->EndTimer(
        "OptimizeNeighborSearch::GetEffectiveBoxesForFeature");

    std::vector<Patch> all_patches;
    for (const auto* box : effective_boxes) {
      Timing::Instance()->StartTimer("OptimizeNeighborSearch::visited_box");
      if (visited_box.count(box) > 0) {
        Timing::Instance()->EndTimer("OptimizeNeighborSearch::visited_box");
        continue;
      }
      visited_box.insert(box);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::visited_box");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::GeneratePatches");

      if (search_mode == SearchMode::ChangeOne) {
        Timing::Instance()->StartTimer("GetNeighborsWithinDistance");
        GetNeighborsWithinDistance(max_dist, feature_id, starting_score,
                                   starting_point, victim_point, box,
                                   layered_box, is_bounded, &all_patches);
        Timing::Instance()->EndTimer("GetNeighborsWithinDistance");
      } else {
        assert(search_mode == SearchMode::NaiveFeature ||
               search_mode == SearchMode::NaiveLeaf);
        GenerateNaivePatches(search_mode, starting_point, box, is_bounded,
                             layered_box->GetCachedIntersection(),
                             &all_patches);
        // DCHECK(patches.back().empty());
        // patches.pop_back();
      }

      Timing::Instance()->EndTimer("OptimizeNeighborSearch::GeneratePatches");
    }

    Timing::Instance()->BinCount("neighbor1t-all_patches.size",
                                 all_patches.size());

    for (const auto& patch : all_patches) {
      Timing::Instance()->BinCount("BeforeHashPatchSize", patch.size());

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::visited_patch");
      if (visited_patch.count(patch) > 0) {
        Timing::Instance()->EndTimer("OptimizeNeighborSearch::visited_patch");
        continue;
      }
      visited_patch.insert(patch);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::visited_patch");

      Timing::Instance()->BinCount("AfterHashPatchSize", patch.size());

      Timing::Instance()->StartTimer(
          "OptimizeNeighborSearch::StretchWithinBox");
      const BoundingBox* constrain_box = nullptr;
      if (search_mode == SearchMode::ChangeOne) {
        constrain_box = patch.box;
      }

      Timing::Instance()->StartTimer(
          "OptimizeNeighborSearch::incompatible_boxes");
      std::vector<const BoundingBox*> incompatible_boxes = GetIncompatibleBoxes(
          patch, layered_box, &cached_incompatible_boxes_per_feature);
      Timing::Instance()->EndTimer(
          "OptimizeNeighborSearch::incompatible_boxes");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::new_boxes");
      std::vector<const BoundingBox*> new_boxes =
          layered_box->GetNewBoxes(patch, incompatible_boxes);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::new_boxes");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::new_adv_scores");
      std::vector<double> new_adv_scores =
          layered_box->GetNewScores(incompatible_boxes, new_boxes);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::new_adv_scores");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::MaxIndex");
      int new_label =
          MaxIndexBetween(new_adv_scores, target_label, victim_label);

      Timing::Instance()->EndTimer("OptimizeNeighborSearch::MaxIndex");

      if (new_label != target_label) {
        Timing::Instance()->BinCount("Filter_Label", 1);
        continue;
      }
      Timing::Instance()->BinCount("Filter_Label", 0);

      // Note: |new_adv_patch| may not have the same bounding box as |patch|
      // before |TightenPoint|.
      Patch new_adv_patch = layered_box->StretchWithinBox(
          patch, victim_point, constrain_box, incompatible_boxes);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::StretchWithinBox");

      // Used stretched |new_adv| and |layered_box->Scores()| to do a quick
      // filter.
      // TODO: Verify if it's ok to use |layered_box->Scores()|.
      if (neighbor_score.Fast(starting_score, starting_point, victim_point,
                              new_adv_patch) < best_score) {
        Timing::Instance()->BinCount("Filter_StretchNorm", 1);

        Timing::Instance()->StartTimer(
            "OptimizeNeighborSearch::incompatible_boxes.clear()");
        incompatible_boxes.clear();
        Timing::Instance()->EndTimer(
            "OptimizeNeighborSearch::incompatible_boxes.clear()");
        continue;
      }
      Timing::Instance()->BinCount("Filter_StretchNorm", 0);

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::new_adv");
      Point new_adv(starting_point);
      new_adv.Apply(new_adv_patch);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::new_adv");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::TightenPoint");
      layered_box->TightenPoint(&new_adv, new_boxes);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::TightenPoint");

      Timing::Instance()->StartTimer(
          "OptimizeNeighborSearch::incompatible_boxes.clear()");
      incompatible_boxes.clear();
      Timing::Instance()->EndTimer(
          "OptimizeNeighborSearch::incompatible_boxes.clear()");

      // TODO: Add back hashing.
      // if (visted_boxes_.count(layered_box->Hash()) > 0)
      //   continue;
      // visted_boxes_.insert(layered_box->Hash());

      // Timing::Instance()->IncreaseSample("VisitsPerBox",
      // layered_box->Hash());

      // TODO: Avoid computing full norm each time.
      Timing::Instance()->StartTimer("OptimizeNeighborSearch::score_func");
      auto new_score =
          neighbor_score.Slow(new_adv, victim_point);
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::score_func");

      Timing::Instance()->StartTimer("OptimizeNeighborSearch::SaveBetter");
      if (new_score > best_score) {
        Timing::Instance()->BinCount("Filter_Score", 0);
        found_better_adv = true;
        best_adv = new_adv;
        best_score = new_score;
        L(best_patch = patch);
      } else {
        Timing::Instance()->BinCount("Filter_Score", 1);
      }
      Timing::Instance()->EndTimer("OptimizeNeighborSearch::SaveBetter");
    }

    if (config_.enable_early_return) {
      ++current_iteration;
      if (found_better_adv && current_iteration % early_return_interval == 0)
        break;
    }
  }

  if (config_.enable_early_return) {
    Timing::Instance()->BinCount(
        "EarlyReturnBatch", (int)(current_iteration / early_return_interval));
  }

  Timing::Instance()->BinCount("bound_trees-visited_box.size()",
                               visited_box.size());
  Timing::Instance()->BinCount("neighbor1-visited_patch.size()",
                               visited_patch.size());

  Timing::Instance()->StartTimer(
      "OptimizeNeighborSearch::visited_patch.clear()");
  visited_patch.clear();
  Timing::Instance()->EndTimer("OptimizeNeighborSearch::visited_patch.clear()");

  Timing::Instance()->StartTimer(
      "OptimizeNeighborSearch::cached_incompatible_boxes_per_feature.clear()");
  cached_incompatible_boxes_per_feature.clear();
  Timing::Instance()->EndTimer(
      "OptimizeNeighborSearch::cached_incompatible_boxes_per_feature.clear()");

  L(cout << "best_patch:" << ToDebugString(best_patch) << endl);

  Timing::Instance()->StartTimer("OptimizeNeighborSearch::ShiftPoint");
  if (found_better_adv)
    layered_box->ShiftPoint(best_adv);
  Timing::Instance()->EndTimer("OptimizeNeighborSearch::ShiftPoint");
  return found_better_adv;
}

std::pair<bool, Point> NeighborAttack::OptimizeRandomSearch(
    const Point& adv_point,
    const Point& victim_point,
    int victim_label) const {
  Timing::Instance()->StartTimer("OptimizeRandomSearch");

  double l_inf = adv_point.Norm(victim_point, -1);
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0, 2 * l_inf);
  std::bernoulli_distribution should_change(0.1);

  bool found_better_adv = false;
  Point best_adv = adv_point;

  const int len = adv_point.Size();
  Point p_min(len), p_max(len);
  double l_inf_box = fabs(l_inf - 3 * eps);
  for (int i = 0; i < len; ++i) {
    p_min[i] = victim_point[i] - l_inf_box;
    p_max[i] = victim_point[i] + l_inf_box;
  }

  int ct_used_trials = 0;

  Timing::Instance()->StartTimer("OptimizeRandomSearch::trial_loop");
  const int kMaxTrial = 300;
  Point p(len);
  while (ct_used_trials < kMaxTrial) {
    ++ct_used_trials;
    for (int i = 0; i < len; ++i) {
      p[i] = Clip(adv_point[i] +
                      distribution(generator) * int(should_change(generator)),
                  p_min[i], p_max[i]);
    }

    Timing::Instance()->StartTimer(
        "OptimizeRandomSearch::trial_loop::PredictLabel");
    // |p| is guaranteed to have smaller norm on l-inf. We have
    // |kMaxRandomFallback| to guard other norms.
    if (!HasSameLabel(forest_->PredictLabel(p), victim_label)) {
      best_adv = p;
      found_better_adv = true;
      break;
    }
    Timing::Instance()->EndTimer(
        "OptimizeRandomSearch::trial_loop::PredictLabel");
  }
  Timing::Instance()->EndTimer("OptimizeRandomSearch::trial_loop");

  Timing::Instance()->BinCount("OptimizeRandomSearch::ct_used_trials",
                               ct_used_trials);

  Timing::Instance()->EndTimer("OptimizeRandomSearch");
  return std::make_pair(found_better_adv, best_adv);
}

std::pair<bool, Point> NeighborAttack::ReverseNeighborSearch(
    const Point& initial_point,
    int target_label,
    int victim_label,
    const BoundingBox& p_constrain) const {
  Timing::Instance()->StartTimer("ReverseNeighborSearch");
  Timing::Instance()->StartTimer("ReverseNeighborSearch::Setup");

  Timing::Instance()->StartTimer(
      "ReverseNeighborSearch::Setup::GetLayeredBoundingBox");
  auto layered_box =
      forest_->GetLayeredBoundingBox(initial_point, target_label, victim_label);
  Timing::Instance()->EndTimer(
      "ReverseNeighborSearch::Setup::GetLayeredBoundingBox");

  Timing::Instance()->StartTimer(
      "ReverseNeighborSearch::Setup::GetBoxForAllTree");
  auto all_box = layered_box->GetBoxForAllTree();
  Timing::Instance()->EndTimer(
      "ReverseNeighborSearch::Setup::GetBoxForAllTree");
  assert(all_box.size() > 0);
  std::default_random_engine generator;

  Timing::Instance()->EndTimer("ReverseNeighborSearch::Setup");

  const int kMaxAllBoxIter = 1;
  bool found_adv = false;
  int all_box_iter = 0;
  int per_box_iter = 0;
  while (all_box_iter < kMaxAllBoxIter) {
    ++all_box_iter;
    bool found_better_box = false;

    Timing::Instance()->StartTimer("ReverseNeighborSearch::shuffle");
    std::shuffle(all_box.begin(), all_box.end(), generator);
    Timing::Instance()->EndTimer("ReverseNeighborSearch::shuffle");

    int len = std::min(100, (int)all_box.size());
    for (int i = 0; i < len; ++i) {
      const auto* box_to_replace = all_box[i];
      if (found_adv)
        break;
      ++per_box_iter;
      double old_label =
          (box_to_replace->OwnerTree()->ClassId() == target_label)
              ? -box_to_replace->Label()
              : box_to_replace->Label();

      Timing::Instance()->StartTimer(
          "ReverseNeighborSearch::GetAlternativeBoxes");
      auto alt_boxes = layered_box->GetAlternativeBoxes(
          BoundingBox(), 1, box_to_replace, config_.enable_relaxed_boundary,
          &p_constrain);
      Timing::Instance()->BinCount("ReverseNeighborSearch::alt_boxes.size()",
                                   alt_boxes.size());
      Timing::Instance()->EndTimer(
          "ReverseNeighborSearch::GetAlternativeBoxes");

      double min_label = old_label;
      const BoundingBox* min_box = nullptr;
      for (const auto* new_box : alt_boxes) {
        double new_label = (new_box->OwnerTree()->ClassId() == target_label)
                               ? -new_box->Label()
                               : new_box->Label();
        if (new_label < min_label) {
          min_label = new_label;
          min_box = new_box;
        }
      }

      if (min_box) {
        found_better_box = true;
        Timing::Instance()->StartTimer("ReverseNeighborSearch::ShiftByPatch");
        auto patch = min_box->ClosestPatchTo(layered_box->Location());
        layered_box->ShiftByPatch(patch);
        Timing::Instance()->EndTimer("ReverseNeighborSearch::ShiftByPatch");
        if (layered_box->PredictionLabel() == target_label) {
          found_adv = true;
          break;
        }
      }
    }

    if (!found_better_box)
      break;
  }

  Timing::Instance()->BinCount("ReverseNeighborSearch::all_box_iter",
                               all_box_iter);
  Timing::Instance()->BinCount("ReverseNeighborSearch::per_box_iter",
                               per_box_iter);

  Timing::Instance()->EndTimer("ReverseNeighborSearch");

  return std::make_pair(found_adv, layered_box->Location());
}

void NeighborAttack::GetNeighborsWithinDistance(
    int max_dist,
    int target_feature_id,
    const ScoreType& starting_score,
    const Point& starting_point,
    const Point& victim_point,
    const BoundingBox* box_to_replace,
    const LayeredBoundingBox* layered_box,
    const std::vector<FeatureDir>& is_bounded,
    std::vector<Patch>* candidate_patches) const {
  Timing::Instance()->StartTimer(
      "GetNeighborsWithinDistance::GetAlternativeBoxes");
  BoundingBox target_feature_constrain;
  Interval inter;
  if (is_bounded[target_feature_id] == FeatureDir::Upper) {
    inter.lower = starting_point[target_feature_id] + 2 * eps;
  } else {
    inter.upper = starting_point[target_feature_id] - 2 * eps;
  }
  target_feature_constrain.IntersectFeature(target_feature_id, inter);
  auto alt_boxes = layered_box->GetAlternativeBoxes(
      target_feature_constrain, max_dist, box_to_replace,
      config_.enable_relaxed_boundary);

  Timing::Instance()->BinCount("alt_boxes.size()", alt_boxes.size());
  Timing::Instance()->EndTimer(
      "GetNeighborsWithinDistance::GetAlternativeBoxes");

  Timing::Instance()->StartTimer(
      "GetNeighborsWithinDistance::candidate_patches");

  for (const auto* box : alt_boxes) {
    Patch alt_patch;
    alt_patch.box = box;
    for (const auto& iter : box->Intervals()) {
      int feature_id = iter.first;
      const Interval& inter = iter.second;
      if (!inter.Contains(starting_point[feature_id]))
        alt_patch[feature_id] = inter.ClosestTo(starting_point[feature_id]);
    }

    if (!alt_patch.empty())
      candidate_patches->emplace_back(std::move(alt_patch));
  }
  Timing::Instance()->EndTimer("GetNeighborsWithinDistance::candidate_patches");
}

double NeighborAttack::Norm(const Point& p1, const Point& p2) const {
  Timing::Instance()->StartTimer("NeighborAttack::Norm");
  double n = p1.Norm(p2, config_.norm_type);
  Timing::Instance()->EndTimer("NeighborAttack::Norm");
  return n;
}

double NeighborAttack::NormFast(const Patch& new_patch,
                                double old_norm,
                                const Point& old_point,
                                const Point& ref_point) const {
  return cz::NormFast(old_norm, old_point, ref_point, new_patch,
                      config_.norm_type);
}

std::pair<bool, Point> NeighborAttack::PureRandomPoint(const Point& ref_point,
                                                       int victim_label,
                                                       int rseed) const {
  Timing::Instance()->StartTimer("PureRandomPoint");

  std::default_random_engine generator(rseed);
  std::uniform_real_distribution<double> distribution(-0.5, 1.5);

  Point p(ref_point.Size());

  int max_try = 3000;
  while (max_try > 0) {
    --max_try;
    for (auto& v : p) {
      v = distribution(generator);
    }

    if (!HasSameLabel(forest_->PredictLabel(p), victim_label)) {
      Timing::Instance()->EndTimer("PureRandomPoint");
      return std::make_pair(true, p);
    }
  }

  return std::make_pair(false, Point());
}

std::pair<bool, Point> NeighborAttack::NormalRandomPoint(int victim_label,
                                                         const Point& ref_point,
                                                         int rseed) const {
  Timing::Instance()->StartTimer("NormalRandomPoint");

  std::default_random_engine generator(rseed);
  std::normal_distribution<double> distribution(0.0, 1.0);

  const int len = ref_point.Size();
  Point p(len);

  int max_try = 300;
  while (max_try > 0) {
    --max_try;
    for (int i = 0; i < len; ++i) {
      p[i] = ref_point[i] + distribution(generator);
    }

    if (!HasSameLabel(forest_->PredictLabel(p), victim_label)) {
      Timing::Instance()->EndTimer("NormalRandomPoint");
      return std::make_pair(true, p);
    }
  }
  Timing::Instance()->EndTimer("NormalRandomPoint");

  printf(
      "NormalRandomPoint failed after 300 trials for victim_label: %d. "
      "fallback to FeatureSplitsRandomPoint.\n",
      victim_label);
  return FeatureSplitsRandomPoint(victim_label, ref_point, rseed);
}

std::pair<bool, Point> NeighborAttack::FeatureSplitsRandomPoint(
    int victim_label,
    const Point& ref_point,
    int rseed) const {
  Timing::Instance()->StartTimer("FeatureSplitsRandomPoint");

  std::default_random_engine generator(rseed);
  std::uniform_int_distribution<int> distribution(0, 30000007);

  const auto& feature_splits = forest_->FeatureSplits();
  const int len = ref_point.Size();
  Point p(len);

  int max_try = 10000;
  while (max_try > 0) {
    --max_try;
    for (int i = 0; i < len; ++i) {
      p[i] = feature_splits[i][distribution(generator) %
                               feature_splits[i].size()] +
             eps;
    }

    if (!HasSameLabel(forest_->PredictLabel(p), victim_label)) {
      Timing::Instance()->EndTimer("FeatureSplitsRandomPoint");
      return std::make_pair(true, p);
    }
  }

  Timing::Instance()->EndTimer("FeatureSplitsRandomPoint");

  return std::make_pair(false, Point());
}

bool NeighborAttack::FlipWithDirection(LayeredBoundingBox* layered_box,
                                       const Patch& dir) const {
  Patch p;
  Timing::Instance()->StartTimer("FlipWithDirection::Loops");
  const auto& intervals = layered_box->GetCachedIntersection()->Intervals();

  for (const auto& iter : dir) {
    int feature_id = iter.first;
    int feature_dir = iter.second;

    const auto& inter = intervals.find(feature_id);
    if (inter == intervals.end())
      continue;

    if (feature_dir < 0 && inter->second.lower.has_value()) {
      p[feature_id] = inter->second.lower.value() - eps;
    } else if (feature_dir > 0 && inter->second.upper.has_value()) {
      p[feature_id] = inter->second.upper.value() + eps;
    }
  }
  Timing::Instance()->EndTimer("FlipWithDirection::Loops");

  Timing::Instance()->StartTimer("FlipWithDirection::ShiftByPatch");
  if (!p.empty())
    layered_box->ShiftByPatch(p);
  Timing::Instance()->EndTimer("FlipWithDirection::ShiftByPatch");
  return !p.empty();
}

void NeighborAttack::UpdateResult(const Point& adv_point,
                                  const Point& victim_point,
                                  Result* result) const {
  Timing::Instance()->StartTimer("NeighborAttack::UpdateResult");

  auto p = (adv_point - victim_point);

  for (const int norm_type : kAllowedNormTypes) {
    double new_norm = p.Norm(norm_type);
    if (new_norm < result->best_norms[norm_type]) {
      result->best_norms[norm_type] = new_norm;
      result->best_points[norm_type] = adv_point;
    }
  }

  Timing::Instance()->EndTimer("NeighborAttack::UpdateResult");
}

std::string NeighborAttack::GetNormStringForLogging(
    const Point& adv_point,
    const Point& victim_point) const {
  auto p = (adv_point - victim_point);

  std::string s;
  for (const int norm_type : kAllowedNormTypes) {
    s += " Norm(" + std::to_string(norm_type) +
         ")=" + std::to_string(p.Norm(norm_type)) + " ";
  }
  return s;
}

}  // namespace cz
