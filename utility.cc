#include "utility.h"

#include <math.h>
#include <fstream>

namespace cz {

FeatureDir& operator|=(FeatureDir& lhs, FeatureDir rhs) {
  lhs = static_cast<FeatureDir>(
      static_cast<std::underlying_type<FeatureDir>::type>(lhs) |
      static_cast<std::underlying_type<FeatureDir>::type>(rhs));
  return lhs;
}

bool operator&(FeatureDir lhs, FeatureDir rhs) {
  return static_cast<bool>(
      static_cast<std::underlying_type<FeatureDir>::type>(lhs) &
      static_cast<std::underlying_type<FeatureDir>::type>(rhs));
}

bool IsEq(double a, double b) {
  return fabs(a - b) < eps;
}

bool IsEq(const std::optional<double>& a, const std::optional<double>& b) {
  if (a.has_value() != b.has_value())
    return false;

  if (!a.has_value())
    return true;

  return IsEq(a.value(), b.value());
}

bool IsEq(const std::string& a, const std::string& b) {
  return a == b;
}

std::vector<std::pair<int, Point>> LoadSVMFile(const char* path,
                                             int feature_dim,
                                             int start_idx) {
  std::vector<std::pair<int, Point>> parsed_data;

  std::ifstream fin(path);
  std::string line;

  while (std::getline(fin, line)) {
    if (line.size() <= 2)
      break;
    auto* s = line.c_str();
    int label;
    int pos;
    sscanf(s, "%d%n", &label, &pos);
    s += pos;
    Point p(feature_dim + start_idx);
    int axis;
    double value;
    while (sscanf(s, "%d:%lf%n", &axis, &value, &pos) == 2) {
      assert(axis < p.Size());
      // assert(value >= 0 && value <= 1.0);
      value = std::max(0.0, std::min(1.0, value));
      p[axis] = value;
      s += pos;
    }
    parsed_data.emplace_back(std::make_pair(label, std::move(p)));
  }

  return std::move(parsed_data);
}

double Clip(double v, double min_v, double max_v) {
  assert(min_v <= max_v);
  return fmax(fmin(v, max_v), min_v);
}

int MaxIndex(const std::vector<double>& v, int except) {
  assert(v.size() > 0);
  int max_index = (except == 0) ? 1 : 0;
  double max_score = v[max_index];
  for (int i = 0; i < v.size(); ++i) {
    if (i == except)
      continue;
    if (v[i] > max_score) {
      max_score = v[i];
      max_index = i;
    }
  }
  return max_index;
}

int MaxIndexBetween(const std::vector<double>& v, int class1, int class2) {
  if (v[class2] > v[class1])
    return class2;
  return class1;
}

Direction ToTrimmedDirection(const Patch& patch, const Point& ref_point) {
  Direction dir;
  for (const auto& iter : patch) {
    if (iter.second == ref_point[iter.first])
      continue;
    if (iter.second > ref_point[iter.first]) {
      dir.push_back(iter.first);
    } else {
      dir.push_back(-iter.first);
    }
  }
  return std::move(dir);
}

double NormFast(double old_norm,
                const Point& old_point,
                const Point& ref_point,
                const Patch& new_patch,
                int norm_type) {
  if (norm_type == 1) {
    for (const auto& feature_value : new_patch) {
      const int feature_id = feature_value.first;
      const double value = feature_value.second;
      old_norm += fabs(value - ref_point[feature_id]) -
                  fabs(old_point[feature_id] - ref_point[feature_id]);
    }
    return old_norm;
  } else if (norm_type == 2) {
    old_norm = std::pow(old_norm, 2);
    for (const auto& feature_value : new_patch) {
      const int feature_id = feature_value.first;
      const double value = feature_value.second;
      old_norm += std::pow(value - ref_point[feature_id], 2) -
                  std::pow(old_point[feature_id] - ref_point[feature_id], 2);
    }
    return sqrt(old_norm);
  } else if (norm_type == -1) {
    double new_diff_max = -1;
    double old_diff_max = -1;

    for (const auto& feature_value : new_patch) {
      const int feature_id = feature_value.first;
      const double value = feature_value.second;
      new_diff_max =
          std::max(new_diff_max, fabs(value - ref_point[feature_id]));
      old_diff_max = std::max(
          old_diff_max, fabs(old_point[feature_id] - ref_point[feature_id]));
    }

    if (new_diff_max >= old_norm)
      return new_diff_max;

    if (old_diff_max < old_norm)
      return old_norm;

    int len = old_point.Size();
    for (int i = 0; i < len; ++i) {
      const double t = fabs(old_point[i] - ref_point[i]);
      if (t > new_diff_max && new_patch.find(i) == new_patch.end())
        new_diff_max = t;
    }
    return new_diff_max;
  }
  cout << "Unsupported norm_type:" << norm_type << endl;
  assert(false);
  return -1;
}

}  // namespace cz
