#pragma once

#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <chrono>
#include <ctime>
#include <map>
#include <memory>
#include <string>

namespace cz {

class Timing {
 public:
  // |int| overflows on FASHION-MNIST.
  using IntType = long long;
  using DoubleType = long double;

  static Timing* Instance() {
    if (!collect_histogram_)
      return dummy_instance_.get();

    // Don't delete p during cleanup.
    static boost::thread_specific_ptr<Timing> tls([](Timing* t) {});
    if (!tls.get()) {
      boost::mutex::scoped_lock l(pti_mtx_);
      auto& p = per_thread_instance_[boost::this_thread::get_id()];
      if (!p.get())
        p = std::make_unique<Timing>();
      tls.reset(p.get());
    }
    return tls.get();
  }

  // Should be diabled during large scale benchmarks.
  void SetCollectHistogram(bool collect_histogram);

  void StartTimer(const char* tag);
  void EndTimer(const char* tag);

  double GetTotalSeconds(const char* tag);

  void BinCount(const char* name, int bin);
  void IncreaseSample(const char* name, size_t sample);
  void IncreaseSample(const char* name, int sample);

  std::string CollectMetricsString();

 private:
  void CollectMetrics();
  void DumpSamplesToBins();
  std::string ToDebugString();

  std::map<const char*, std::chrono::high_resolution_clock::time_point>
      start_time_;
  std::map<const char*, std::pair<IntType, DoubleType>>
      timer_count_total_seconds_;

  std::map<const char*, std::map<IntType, IntType>> bins_;
  std::map<const char*, std::map<IntType, IntType>> samples_;

  static bool collect_histogram_;

  static boost::mutex pti_mtx_;
  static std::map<boost::thread::id, std::unique_ptr<Timing>>
      per_thread_instance_;

  // Only used when |collect_histogram_| is false;
  static std::unique_ptr<Timing> dummy_instance_;
};

}  // namespace cz
