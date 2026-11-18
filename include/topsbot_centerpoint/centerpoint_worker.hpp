#pragma once

#include <topsbot_centerpoint/centerpoint.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace centerpoint {

// Load balancing strategy for multi-core execution
enum class BalanceStrategy {
  LeastLoaded,   // Assign to worker with least in-flight requests
  RoundRobin     // Cycle through workers in order
};

struct WorkerStats {
  bool healthy{false};
  std::size_t in_flight{0};
  uint64_t assigned{0};
  uint64_t completed{0};
  uint64_t failed{0};
  uint64_t dropped{0};
  double latency_ewma_ms{0.0};  // Exponential weighted moving average
};

// Per-worker detection request
struct DetectionRequest {
  PillarBatch pillars;
  std::chrono::steady_clock::time_point enqueued_at;
  uint64_t frame_id{0};
  // Opaque caller context (e.g. BEV render snapshot) forwarded to the result.
  std::shared_ptr<void> context;
};

struct DetectionResult {
  std::vector<Detection> detections;
  Timing timing;
  uint64_t frame_id{0};
  bool success{false};
  std::string error;
  int device_id{-1};
  std::shared_ptr<void> context;
};

// Thread-safe load balancer for distributing work across NPU cores
class LoadBalancer {
 public:
  explicit LoadBalancer(std::size_t worker_count,
                        BalanceStrategy strategy = BalanceStrategy::LeastLoaded);

  std::optional<std::size_t> acquire();
  void complete(std::size_t worker_index, bool success, double latency_ms);
  void cancel(std::size_t worker_index);
  void set_healthy(std::size_t worker_index, bool healthy);

  std::vector<WorkerStats> snapshot() const;
  std::size_t worker_count() const noexcept { return workers_.size(); }
  BalanceStrategy strategy() const noexcept { return strategy_; }

 private:
  std::optional<std::size_t> choose_round_robin_locked();
  std::optional<std::size_t> choose_least_loaded_locked() const;

  mutable std::mutex mutex_;
  std::vector<WorkerStats> workers_;
  BalanceStrategy strategy_;
  std::size_t round_robin_cursor_{0};
};

// Single worker thread managing one NPU core
class Worker {
 public:
  Worker(int device_id, const std::string& pfe_model,
         const std::string& rpn_model, const std::string& merged_model = "");
  ~Worker();

  bool initialize(std::string* error, float score_threshold = -1.0f);
  void start(std::size_t worker_index, LoadBalancer* balancer);
  void stop();
  bool enqueue(DetectionRequest request, std::size_t max_queue_size);

  int device_id() const noexcept { return device_id_; }
  bool is_ready() const noexcept { return ready_; }

  // Result callback: called from worker thread after each detection
  void set_result_callback(
      std::function<void(const DetectionResult&)> callback) {
    result_callback_ = std::move(callback);
  }

 private:
  void worker_loop();

  int device_id_;
  std::string pfe_model_;
  std::string rpn_model_;
  std::string merged_model_;

  CenterPoint detector_;
  bool ready_{false};

  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<DetectionRequest> queue_;
  bool stop_{false};
  std::thread thread_;

  std::size_t worker_index_{0};
  LoadBalancer* balancer_{nullptr};
  std::function<void(const DetectionResult&)> result_callback_;
};

// Multi-core CenterPoint manager
class CenterPointWorkerPool {
 public:
  struct Config {
    std::vector<int> device_ids{0};  // NPU cores to use
    std::string pfe_model;
    std::string rpn_model;
    std::string merged_model;  // single-graph NB (PFE+ScatterND+RPN); wins over pfe/rpn
    std::size_t max_queue_size{2};
    BalanceStrategy strategy{BalanceStrategy::LeastLoaded};
    float score_threshold{0.2f};
  };

  explicit CenterPointWorkerPool(const Config& config);
  ~CenterPointWorkerPool();

  bool initialize(std::string* error);

  // Async detection: returns immediately, result delivered via callback
  bool submit(DetectionRequest request);

  // Set callback for detection results
  void set_result_callback(
      std::function<void(const DetectionResult&)> callback);

  // Get load balancer stats
  std::vector<WorkerStats> get_stats() const;

  std::size_t worker_count() const noexcept { return workers_.size(); }
  bool is_initialized() const noexcept { return initialized_; }

 private:
  Config config_;
  std::vector<std::unique_ptr<Worker>> workers_;
  std::unique_ptr<LoadBalancer> load_balancer_;
  std::function<void(const DetectionResult&)> result_callback_;
  bool initialized_{false};
  std::atomic<uint64_t> frame_counter_{0};
};

}  // namespace centerpoint
