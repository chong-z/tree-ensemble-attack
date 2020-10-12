#include "interval.h"

#include <assert.h>
#include <algorithm>

#include "utility.h"

namespace cz {

namespace {

std::string ToString(const std::optional<double>& d) {
  if (!d.has_value())
    return "INF";
  return std::to_string(d.value());
}

}  // namespace

Interval Interval::Lower(double other_lower) {
  Interval interval;
  interval.lower = other_lower;
  return interval;
}

Interval Interval::Upper(double other_upper) {
  Interval interval;
  interval.upper = other_upper;
  return interval;
}

void Interval::IntersectLower(double other_lower) {
  lower = std::max(other_lower, lower.value_or(other_lower));
  assert(!upper.has_value() || lower.value() <= upper.value());
}

void Interval::IntersectUpper(double other_upper) {
  upper = std::min(other_upper, upper.value_or(other_upper));
  assert(!lower.has_value() || lower.value() <= upper.value());
}

void Interval::Intersect(const Interval& other) {
  if (other.lower.has_value())
    IntersectLower(other.lower.value());

  if (other.upper.has_value())
    IntersectUpper(other.upper.value());
}

bool Interval::HasSameUpper(const Interval& rhs) const {
  if (!upper.has_value() || !rhs.upper.has_value())
    return false;

  return upper.value() == rhs.upper.value();
}

bool Interval::HasSameLower(const Interval& rhs) const {
  if (!lower.has_value() || !rhs.lower.has_value())
    return false;

  return lower.value() == rhs.lower.value();
}

bool Interval::operator==(const Interval& rhs) const {
  return lower == rhs.lower && upper == rhs.upper;
}

bool Interval::HasStricterUpper(const Interval& rhs) const {
  if (!upper.has_value())
    return false;

  return !rhs.upper.has_value() || upper.value() <= rhs.upper.value();
}

bool Interval::HasStricterLower(const Interval& rhs) const {
  if (!lower.has_value())
    return false;

  return !rhs.lower.has_value() || lower.value() >= rhs.lower.value();
}

bool Interval::HasValue() const {
  return lower.has_value() || upper.has_value();
}

bool Interval::Contains(double v) const {
  if (lower.has_value() && v < lower.value())
    return false;

  if (upper.has_value() && v >= upper.value())
    return false;

  return true;
}

bool Interval::Overlaps(const Interval& other) const {
  if (lower.value_or(-DBL_MAX) >= other.upper.value_or(DBL_MAX))
    return false;

  if (other.lower.value_or(-DBL_MAX) >= upper.value_or(DBL_MAX))
    return false;

  return true;
}

bool Interval::Adjacents(const Interval& other) const {
  return lower.value_or(-DBL_MAX) == other.upper.value_or(DBL_MAX) ||
         other.lower.value_or(-DBL_MAX) == upper.value_or(DBL_MAX);
}

double Interval::ClosestTo(double v) const {
  return fmax(fmin(v, upper.value_or(DBL_MAX) - eps),
              lower.value_or(-DBL_MAX) + eps);
}

std::string Interval::ToDebugString() const {
  return std::string("(") + ToString(lower) + "," + ToString(upper) + ")";
}

}  // namespace cz
