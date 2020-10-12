#pragma once

#include <optional>
#include <string>

namespace cz {

struct Interval {
  static Interval Lower(double other_lower);
  static Interval Upper(double other_upper);

  void IntersectLower(double other_lower);
  void IntersectUpper(double other_upper);
  void Intersect(const Interval&);

  // Both |has_value()| && same |value()|.
  bool HasSameUpper(const Interval&) const;
  bool HasSameLower(const Interval&) const;

  bool operator==(const Interval&) const;

  bool HasStricterUpper(const Interval&) const;
  bool HasStricterLower(const Interval&) const;

  bool HasValue() const;
  bool Contains(double v) const;

  // Has non-empty intersection.
  bool Overlaps(const Interval&) const;
  // Has empty intersection but the same lower-upper.
  bool Adjacents(const Interval&) const;
  double ClosestTo(double v) const;

  std::string ToDebugString() const;

  std::optional<double> lower;
  std::optional<double> upper;
};

}  // namespace cz
