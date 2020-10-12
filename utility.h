#pragma once

#include <assert.h>
#include <limits.h>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

using std::cout;
using std::endl;

namespace cz {

class BoundingBox;

// TODO: Verify if it's small enough for all datasets.
const double eps = 1e-10;

#ifdef DEBUG
#define DCHECK(condition) \
  if (!(condition))       \
  asm("int $3")
#else
#define DCHECK(condition) \
  do {                    \
  } while (0)
#endif

#ifdef DEBUG
#define L(condition) condition
#else
#define L(condition) \
  do {               \
  } while (0)
#endif

// using Patch = std::map<int, double>;

class Patch : public std::map<int, double> {
 public:
  Patch() {}
  Patch(std::initializer_list<std::pair<const int, double>> l)
      : std::map<int, double>(l) {}
  Patch(const Patch&) = default;
  Patch(Patch&&) = default;
  Patch& operator=(const Patch&) = default;

  const BoundingBox* box = nullptr;
};

struct PatchCompare {
  bool operator()(const Patch& lhs, const Patch& rhs) const {
    if (lhs.size() != rhs.size())
      return lhs.size() < rhs.size();

    auto l = lhs.begin();
    auto end = lhs.end();
    auto r = rhs.begin();
    while (l != end) {
      if (l->first != r->first)
        return l->first < r->first;
      if (l->second != r->second)
        return l->second < r->second;
      ++l;
      ++r;
    }

    return false;
  }
};

// positive: to upper; negative: to lower.
using Direction = std::vector<int>;

enum class SearchMode {
  // Search all neighbors with hamming dist 1.
  ChangeOne,
  // A naive method that enumerates all leaves.
  NaiveLeaf,
  // A naive method that flips one feature at a time.
  NaiveFeature,
  // The Region-Based Attack Approx. (Yang et al. 2019).
  Region,
};

enum class FeatureDir : unsigned {
  None = 0,
  Lower = 1u << 0,
  Upper = 1u << 1,
  Both = Lower & Upper
};

FeatureDir& operator|=(FeatureDir& lhs, FeatureDir rhs);
bool operator&(FeatureDir lhs, FeatureDir rhs);

template <class T>
T GetOr(const nlohmann::json& config, const char* key, T default_value) {
  if (config.find(key) == config.end())
    return default_value;
  return config[key];
}

struct Config {
  Config() {}
  explicit Config(const char* config_path_in) {
    std::ifstream fin(config_path_in);
    nlohmann::json config_json;
    fin >> config_json;
    cout << "Parsing config:" << config_json << endl;
    config_path = config_path_in;
    inputs_path = config_json["inputs"];
    train_path = GetOr<std::string>(config_json, "train_data", "");
    model_path = config_json["model"];
    offset = GetOr(config_json, "offset", 0);
    num_point = config_json["num_point"];
    num_attack_per_point = config_json["num_attack_per_point"];
    num_classes = config_json["num_classes"];
    num_features = config_json["num_features"];

    milp_adv = GetOr<std::string>(config_json, "milp_adv", "");
    adv_training_path =
        GetOr<std::string>(config_json, "adv_training_path", "");
    collect_histogram = GetOr(config_json, "collect_histogram", false);
    enable_relaxed_boundary =
        GetOr(config_json, "enable_relaxed_boundary", false);
    enable_early_return = GetOr(config_json, "enable_early_return", true);
    feature_start = GetOr(config_json, "feature_start", 1);
    norm_type = GetOr(config_json, "norm_type", 2);
    max_dist = GetOr(config_json, "max_dist", 1);
    num_threads = GetOr(config_json, "num_threads", 4);
    binary_search_threshold =
        GetOr(config_json, "binary_search_threshold", 0.01);
    norm_weight = GetOr<double>(config_json, "norm_weight", 0.99999);

    std::string mode_str =
        GetOr<std::string>(config_json, "search_mode", "lt-attack");
    if (mode_str == "lt-attack") {
      search_mode = SearchMode::ChangeOne;
    } else if (mode_str == "naive-leaf") {
      search_mode = SearchMode::NaiveLeaf;
    } else if (mode_str == "naive-feature") {
      search_mode = SearchMode::NaiveFeature;
    } else if (mode_str == "region") {
      search_mode = SearchMode::Region;
      assert(train_path.length() > 0);
    } else {
      assert(false);
    }

    assert(norm_weight >= 0 && norm_weight <= 1);
    assert(binary_search_threshold >= 0 && binary_search_threshold <= 1);
    assert(norm_type == -1 || norm_type == 1 || norm_type == 2);
    assert(feature_start == 0 || feature_start == 1);
  }

  std::string config_path;
  std::string inputs_path;
  std::string train_path;
  std::string model_path;
  std::string milp_adv;
  std::string adv_training_path;
  int offset;
  int num_point;
  int num_attack_per_point;
  int num_classes;
  int num_features;
  int norm_type;
  int max_dist;
  int feature_start;
  int num_threads;
  double binary_search_threshold;
  double norm_weight;
  bool collect_histogram;
  bool enable_relaxed_boundary;
  bool enable_early_return;
  SearchMode search_mode;
};

class Point {
 public:
  Point() {}
  explicit Point(int len) : data_(len) {}
  Point(Point&& rhs) : data_(std::move(rhs.data_)) {}
  Point(const Point& rhs) = default;
  Point(std::initializer_list<double> l) : data_(l) {}
  ~Point() {}

  static Point FromDebugString(const char* s) {
    Point p;
    int pos;
    // Skip first '('.
    s += 1;
    double v;
    while (sscanf(s, "%lf,%n", &v, &pos) >= 1) {
      p.data_.push_back(v);
      s += pos;
    }
    return std::move(p);
  }

  Point& operator=(Point&& rhs) {
    data_ = std::move(rhs.data_);
    return *this;
  }

  Point& operator=(const Point& rhs) = default;

  Point& operator-=(const Point& rhs) {
    assert(data_.size() == rhs.data_.size());
    for (int i = 0; i < data_.size(); ++i)
      data_[i] -= rhs.data_[i];
    return *this;
  }
  Point& operator+=(const Point& rhs) {
    assert(data_.size() == rhs.data_.size());
    for (int i = 0; i < data_.size(); ++i)
      data_[i] += rhs.data_[i];
    return *this;
  }
  Point& operator*=(double rhs) {
    for (int i = 0; i < data_.size(); ++i)
      data_[i] *= rhs;
    return *this;
  }
  Point& operator/=(double rhs) {
    for (int i = 0; i < data_.size(); ++i)
      data_[i] /= rhs;
    return *this;
  }

  Point operator+(const Point& rhs) const {
    assert(data_.size() == rhs.data_.size());
    Point tmp(*this);
    tmp += rhs;
    return std::move(tmp);
  }
  Point operator-(const Point& rhs) const {
    assert(data_.size() == rhs.data_.size());
    Point tmp(*this);
    tmp -= rhs;
    return std::move(tmp);
  }
  Point operator*(double rhs) const {
    Point tmp(*this);
    tmp *= rhs;
    return std::move(tmp);
  }
  Point operator/(double rhs) const {
    Point tmp(*this);
    tmp /= rhs;
    return std::move(tmp);
  }

  double& operator[](int i) {
    assert(i < data_.size());
    return data_[i];
  }

  double operator[](int i) const {
    assert(i < data_.size());
    return data_[i];
  }

  std::vector<double>::iterator begin() { return data_.begin(); }
  std::vector<double>::iterator end() { return data_.end(); }

  void Apply(const Patch& other) {
    for (const auto& iter : other) {
      data_[iter.first] = iter.second;
    }
  }

  Patch Diff(const Point& other) const {
    Patch p;
    int len = data_.size();
    for (int i = 0; i < len; ++i) {
      if (data_[i] != other[i])
        p[i] = data_[i];
    }
    return std::move(p);
  }

  bool empty() const { return data_.empty(); }
  int Size() const { return data_.size(); }

  double Norm(int norm_type) const {
    if (norm_type == 1) {
      double n = 0;
      for (auto v : data_)
        n += fabs(v);
      return n;
    }

    if (norm_type == 2) {
      double n = 0;
      for (auto v : data_)
        n += v * v;
      return sqrt(n);
    }

    if (norm_type == -1) {
      double maxn = -1;
      for (auto v : data_)
        maxn = fmax(maxn, fabs(v));
      return maxn;
    }

    cout << "Unsupported norm_type:" << norm_type << endl;
    assert(false);
    return -1;
  }

  double Norm(const Point& other, int norm_type) const {
    if (norm_type == 1) {
      double n = 0;
      for (int i = 0; i < data_.size(); ++i)
        n += fabs(data_[i] - other[i]);
      return n;
    }

    if (norm_type == 2) {
      double n = 0;
      for (int i = 0; i < data_.size(); ++i)
        n += std::pow(data_[i] - other[i], 2);
      return sqrt(n);
    }

    if (norm_type == -1) {
      double maxn = -1;
      for (int i = 0; i < data_.size(); ++i)
        maxn = fmax(maxn, fabs(data_[i] - other[i]));
      return maxn;
    }

    cout << "Unsupported norm_type:" << norm_type << endl;
    assert(false);
    return -1;
  }

  std::string ToDebugString() const {
    std::ostringstream ss;
    ss << std::fixed;
    ss << std::setprecision(11);
    for (int i = 0; i < data_.size(); ++i) {
      if (data_[i] != 0.0) {
        if (!ss.str().empty())
          ss << " ";
        ss << i << ":" << data_[i];
      }
    }
    return ss.str();
  }

  long long Hash() const {
    long long h = 0;
    for (auto v : data_)
      h = (h << 2) ^ static_cast<long long>(v) ^ (h >> 16);
    return h;
  }

 private:
  std::vector<double> data_;
};

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  TypeName& operator=(const TypeName&) = delete

bool IsEq(double a, double b);
bool IsEq(const std::optional<double>& a, const std::optional<double>& b);
bool IsEq(const std::string& a, const std::string& b);

template <class k, class v>
std::string ToDebugString(const std::map<k, v>& m) {
  std::string s = "{";
  for (const auto& iter : m) {
    s += std::to_string(iter.first) + ": " + std::to_string(iter.second) + ", ";
  }
  s += "}";
  return s;
}

template <class v>
std::string ToDebugString(const std::vector<v>& m) {
  std::string s = "[";
  for (int i = 0; i < m.size(); ++i) {
    s += std::to_string(i) + ": " + std::to_string(m[i]) + ", ";
  }
  s += "]";
  return s;
}

struct DirectionHash {
  // Good for direction with len 2~4, and less than 2^10 features.
  size_t operator()(const Direction& dir) const {
    std::size_t seed = dir.size();
    for (auto i : dir) {
      seed = (seed << 10) + i;
    }
    return seed;
  }
};

std::vector<std::pair<int, Point>> LoadSVMFile(const char* path,
                                             int feature_dim,
                                             int start_idx);

int MaxIndex(const std::vector<double>& v, int except = -1);

int MaxIndexBetween(const std::vector<double>& v, int class1, int class2);

double Clip(double v, double min_v, double max_v);

template <class T>
T Max(const std::vector<T>& v) {
  return *std::max_element(std::begin(v), std::end(v));
}

template <class T>
T Median(const std::vector<T>& v) {
  auto sv(v);
  std::sort(std::begin(sv), std::end(sv));
  return sv[sv.size() / 2];
}

template <class T>
double Mean(const std::vector<T>& v) {
  return (double)std::accumulate(std::begin(v), std::end(v), 0) / v.size();
}

Direction ToTrimmedDirection(const Patch& patch, const Point& ref_point);

double NormFast(double old_norm,
                const Point& old_point,
                const Point& ref_point,
                const Patch& new_patch,
                int norm_type);
}  // namespace cz
