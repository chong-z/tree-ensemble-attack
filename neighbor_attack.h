#pragma once

#include <functional>
#include <memory>

#include "utility.h"

namespace cz {

class BoundingBox;
class DecisionForest;
class LayeredBoundingBox;

// Larger score is better.
using ScoreType = std::pair<double, double>;

class NeighborScore {
 public:
  virtual ScoreType Slow(const Point& point,
                         const Point& ref_point) const = 0;

  virtual ScoreType Fast(const ScoreType& old_score,
                         const Point& old_point,
                         const Point& ref_point,
                         const Patch& new_patch) const = 0;
};

class NeighborAttack {
 public:
  static const std::vector<int> kAllowedNormTypes;

  struct Result {
    std::map<int, double> best_norms;
    std::map<int, Point> best_points;
    std::vector<Point> hist_points;

    Result() {
      for (const int norm_type : kAllowedNormTypes) {
        best_norms[norm_type] = std::numeric_limits<double>::max();
      }
    }

    std::string ToNormString() {
      std::string str;
      for (const int norm_type : kAllowedNormTypes) {
        str += "Norm(" + std::to_string(norm_type) +
               ")=" + std::to_string(best_norms[norm_type]) + " ";
      }
      return str;
    }

    std::string ToPointString() {
      std::string str;
      for (const int norm_type : kAllowedNormTypes) {
        str += "Point(" + std::to_string(norm_type) +
               ")=" + best_points[norm_type].ToDebugString() + "\n";
      }
      return str;
    }

    bool success() {
      for (const int norm_type : kAllowedNormTypes) {
        if (best_norms[norm_type] < std::numeric_limits<double>::max())
          return true;
      }
      return false;
    }
  };

  explicit NeighborAttack(const Config&);
  ~NeighborAttack();

  void LoadForestFromJson(const std::string& path);

  Result FindAdversarialPoint(const Point& victim_point) const;

  int PredictLabel(const Point& point) const;

  int HammingDistanceBetween(const Point& p1, const Point& p2, int class1, int class2) const;
  int NeighborDistanceBetween(const Point& start_point,
                              const Point& end_point,
                              int adv_class,
                              int victim_class,
                              const Point& victim_point) const;

  const DecisionForest* ForestForTesting() const;

 private:
  friend class NeighborAttackTest;

  void FindAdversarialPoint_ThreadRun(int task_id,
                                      const Point& victim_point,
                                      int victim_label,
                                      Result* result) const;

  void NeighborDistanceBetween_ThreadRun(int task_id,
                                         const Point& start_point,
                                         const Point& end_point,
                                         int adv_class,
                                         int victim_class,
                                         const Point& victim_point,
                                         int* out_neighbor_dist) const;

  void RegionBasedAttackAppr_ThreadRun(int task_id,
                                       const Point& victim_point,
                                       int victim_label,
                                       Result* result) const;

  // Start with an |adv_point| and continously move it closer to |victim_point|.
  Point OptimizeAdversarialPoint(Point adv_point,
                                 const Point& victim_point,
                                 int target_label,
                                 int victim_label,
                                 Result* result) const;
  Point OptimizeLocalSearch(const BoundingBox* box,
                            const Point& victim_point) const;
  Point OptimizeBinarySearch(const Point& adv_point,
                             const Point& victim_point,
                             int target_label,
                             int victim_label) const;
  // bool OptimizeSingleNeighborSearch(LayeredBoundingBox* layered_box,
  //                                   const Point& victim_point,
  //                                   int victim_label) const;
  bool OptimizeNeighborSearch(LayeredBoundingBox* layered_box,
                              const Point& victim_point,
                              int target_label,
                              int victim_label,
                              const NeighborScore& neighbor_score,
                              SearchMode search_mode,
                              int max_dist) const;

  std::pair<bool, Point> OptimizeRandomSearch(const Point& adv_point,
                                              const Point& victim_point,
                                              int victim_label) const;

  std::pair<bool, Point> ReverseNeighborSearch(
      const Point& initial_point,
      int target_label,
      int victim_label,
      const BoundingBox& p_constrain) const;

  // The resulting patch must have better norm on |target_feature_id|.
  void GetNeighborsWithinDistance(int max_dist,
                                  int target_feature_id,
                                  const ScoreType& starting_score,
                                  const Point& starting_point,
                                  const Point& victim_point,
                                  const BoundingBox* box_to_replace,
                                  const LayeredBoundingBox* layered_box,
                                  const std::vector<FeatureDir>& is_bounded,
                                  std::vector<Patch>* candidate_patches) const;

  double Norm(const Point& p1, const Point& p2) const;
  double NormFast(const Patch& new_patch,
                  double old_norm,
                  const Point& old_point,
                  const Point& ref_point) const;
  std::pair<bool, Point> PureRandomPoint(const Point& ref_point,
                                         int victim_label,
                                         int rseed) const;
  std::pair<bool, Point> NormalRandomPoint(int victim_label,
                                           const Point& ref_point,
                                           int rseed) const;
  std::pair<bool, Point> FeatureSplitsRandomPoint(int victim_label,
                                                  const Point& ref_point,
                                                  int rseed) const;
  // std::optional<Point> GenFromVictim(const Point& victim_point,
  //                                    int victim_label,
  //                                    double norm_weight) const;
  bool FlipWithDirection(LayeredBoundingBox* layered_box,
                         const Patch& dir) const;

  void ResetAttackCache() const;
  void UpdateResult(const Point& adv_point,
                    const Point& victim_point,
                    Result* result) const;
  std::string GetNormStringForLogging(const Point& adv_point,
                                      const Point& victim_point) const;

  Config config_;

  std::unique_ptr<DecisionForest> forest_;

  // Only for RBA-Appr (Yang et al. 2019).
  std::vector<std::pair<int, Point>> train_data_;

  DISALLOW_COPY_AND_ASSIGN(NeighborAttack);
};

}  // namespace cz
