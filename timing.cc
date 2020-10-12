#include "timing.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace cz {

using namespace std::chrono;

bool Timing::collect_histogram_ = false;

boost::mutex Timing::pti_mtx_;

std::map<boost::thread::id, std::unique_ptr<Timing>>
    Timing::per_thread_instance_;

std::unique_ptr<Timing> Timing::dummy_instance_ = std::make_unique<Timing>();

void Timing::SetCollectHistogram(bool collect_histogram) {
  collect_histogram_ = collect_histogram;
}

void Timing::StartTimer(const char* tag) {
  if (!collect_histogram_)
    return;

  auto& iter = start_time_[tag];
  iter = high_resolution_clock::now();
}

void Timing::EndTimer(const char* tag) {
  if (!collect_histogram_)
    return;

  auto end = high_resolution_clock::now();
  timer_count_total_seconds_[tag].first++;
  timer_count_total_seconds_[tag].second +=
      duration_cast<duration<double>>(end - start_time_[tag]).count();
}

double Timing::GetTotalSeconds(const char* tag) {
  return timer_count_total_seconds_[tag].second;
}

void Timing::BinCount(const char* name, int bin) {
  if (!collect_histogram_)
    return;

  bins_[name][bin]++;
}

void Timing::IncreaseSample(const char* name, size_t sample) {
  if (!collect_histogram_)
    return;

  IncreaseSample(name, (int)sample);
}

void Timing::IncreaseSample(const char* name, int sample) {
  if (!collect_histogram_)
    return;

  samples_[name][sample]++;
}

std::string Timing::CollectMetricsString() {
  if (!collect_histogram_)
    return "|collect_histogram| disabled\n";

  CollectMetrics();
  return ToDebugString();
}

void Timing::CollectMetrics() {
  auto this_id = boost::this_thread::get_id();
  Timing* this_instance = Instance();

  printf("Timing::CollectMetrics per_thread_instance_.size(): %ld\n",
         per_thread_instance_.size());

  for (auto& iter : per_thread_instance_) {
    iter.second->DumpSamplesToBins();
  }

  for (const auto& iter : per_thread_instance_) {
    if (iter.first == this_id)
      continue;

    for (const auto& jter : iter.second->timer_count_total_seconds_) {
      const auto& tag = jter.first;
      timer_count_total_seconds_[tag].first += jter.second.first;
      timer_count_total_seconds_[tag].second += jter.second.second;
    }

    for (const auto& jter : iter.second->bins_) {
      const auto& tag = jter.first;
      for (const auto& kter : jter.second)
        bins_[tag][kter.first] += kter.second;
    }
  }
}

void Timing::DumpSamplesToBins() {
  for (const auto& iter : samples_) {
    for (const auto& jter : iter.second) {
      BinCount(iter.first, jter.second);
    }
  }
  samples_.clear();
}

std::string Timing::ToDebugString() {
  std::string str;

  std::vector<std::pair<std::string, std::pair<IntType, DoubleType>>>
      sorted_time;

  for (const auto& iter : timer_count_total_seconds_) {
    sorted_time.push_back(iter);
  }

  sort(sorted_time.begin(), sorted_time.end());

  for (const auto& iter : sorted_time) {
    str += iter.first + ": " + std::to_string(iter.second.first) + " timers " +
           std::to_string(iter.second.second) + " seconds\n";
  }

  str += "Bins:\n";
  for (const auto& iter : bins_) {
    DoubleType sum = 0;
    IntType num = 0;
    std::string str2 = "{";
    for (const auto& jter : iter.second) {
      str2 += std::to_string(jter.first) + ": " + std::to_string(jter.second) +
              ", ";
      num += jter.second;
      sum += jter.first * jter.second;
    }
    str2 += "}\n";
    str += std::string(iter.first) + ": mean(" + std::to_string(sum / num) +
           ") " + str2;
  }
  return str;
}

}  // namespace cz
