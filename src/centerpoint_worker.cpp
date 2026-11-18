#include <topsbot_centerpoint/centerpoint_worker.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>

namespace centerpoint {

namespace {
constexpr double kEwmaAlpha = 0.2;  // Exponential weighted moving average factor
}

// ============================================================================
// LoadBalancer implementation
// ============================================================================

LoadBalancer::LoadBalancer(std::size_t worker_count, BalanceStrategy strategy)
    : workers_(worker_count), strategy_(strategy) {}

std::optional<std::size_t> LoadBalancer::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);

  std::optional<std::size_t> selected;
  if (strategy_ == BalanceStrategy::RoundRobin) {
    selected = choose_round_robin_locked();
  } else {
    selected = choose_least_loaded_locked();
  }

  if (selected.has_value()) {
    auto& worker = workers_[*selected];
    ++worker.in_flight;
    ++worker.assigned;
  }
  return selected;
}

void LoadBalancer::complete(std::size_t worker_index, bool success,
                            double latency_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_index >= workers_.size()) return;

  auto& worker = workers_[worker_index];
  if (worker.in_flight > 0) --worker.in_flight;

  if (success) {
    ++worker.completed;
    // Update exponential weighted moving average
    if (worker.latency_ewma_ms == 0.0) {
      worker.latency_ewma_ms = latency_ms;
    } else {
      worker.latency_ewma_ms =
          kEwmaAlpha * latency_ms + (1.0 - kEwmaAlpha) * worker.latency_ewma_ms;
    }
  } else {
    ++worker.failed;
  }
}

void LoadBalancer::cancel(std::size_t worker_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_index >= workers_.size()) return;

  auto& worker = workers_[worker_index];
  if (worker.in_flight > 0) --worker.in_flight;
  ++worker.dropped;
}

void LoadBalancer::set_healthy(std::size_t worker_index, bool healthy) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (worker_index < workers_.size()) {
    workers_[worker_index].healthy = healthy;
  }
}

std::vector<WorkerStats> LoadBalancer::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_;
}

std::optional<std::size_t> LoadBalancer::choose_round_robin_locked() {
  const std::size_t start = round_robin_cursor_;
  for (std::size_t attempt = 0; attempt < workers_.size(); ++attempt) {
    const std::size_t index = (start + attempt) % workers_.size();
    if (workers_[index].healthy) {
      round_robin_cursor_ = (index + 1) % workers_.size();
      return index;
    }
  }
  return std::nullopt;
}

std::optional<std::size_t> LoadBalancer::choose_least_loaded_locked() const {
  std::optional<std::size_t> best;
  std::size_t min_load = std::numeric_limits<std::size_t>::max();

  for (std::size_t i = 0; i < workers_.size(); ++i) {
    if (workers_[i].healthy && workers_[i].in_flight < min_load) {
      min_load = workers_[i].in_flight;
      best = i;
    }
  }
  return best;
}

// ============================================================================
// Worker implementation
// ============================================================================

Worker::Worker(int device_id, const std::string& pfe_model,
               const std::string& rpn_model, const std::string& merged_model)
    : device_id_(device_id),
      pfe_model_(pfe_model),
      rpn_model_(rpn_model),
      merged_model_(merged_model) {}

Worker::~Worker() {
  stop();
}

bool Worker::initialize(std::string* error, float score_threshold) {
  if (!merged_model_.empty()) {
    ready_ = detector_.InitMerged(merged_model_, static_cast<uint32_t>(device_id_));
    if (!ready_ && error) {
      *error = "CenterPoint InitMerged failed on NPU core " + std::to_string(device_id_);
    }
  } else {
    ready_ = detector_.Init(pfe_model_, rpn_model_,
                            static_cast<uint32_t>(device_id_));
    if (!ready_ && error) {
      *error = "CenterPoint Init failed on NPU core " + std::to_string(device_id_);
    }
  }
  if (ready_ && score_threshold >= 0.0f) {
    detector_.set_score_threshold(score_threshold);
  }
  return ready_;
}

void Worker::start(std::size_t worker_index, LoadBalancer* balancer) {
  worker_index_ = worker_index;
  balancer_ = balancer;

  if (ready_ && !thread_.joinable()) {
    thread_ = std::thread(&Worker::worker_loop, this);
  }
}

void Worker::stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;

    // Drain queue and notify balancer
    while (!queue_.empty()) {
      if (balancer_) {
        balancer_->cancel(worker_index_);
      }
      queue_.pop_front();
    }
  }
  condition_.notify_all();

  if (thread_.joinable()) {
    thread_.join();
  }

  detector_.Deinit();
}

bool Worker::enqueue(DetectionRequest request, std::size_t max_queue_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (stop_ || !ready_ || queue_.size() >= max_queue_size) {
    return false;
  }

  queue_.push_back(std::move(request));
  condition_.notify_one();
  return true;
}

void Worker::worker_loop() {
  while (true) {
    DetectionRequest request;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this]() { return stop_ || !queue_.empty(); });

      if (stop_ && queue_.empty()) {
        return;
      }

      request = std::move(queue_.front());
      queue_.pop_front();
    }

    const auto started_at = std::chrono::steady_clock::now();

    DetectionResult result;
    result.frame_id = request.frame_id;
    result.device_id = device_id_;
    result.context = request.context;

    try {
      result.success = detector_.Detect(request.pillars, &result.detections,
                                       &result.timing);
      if (!result.success) {
        result.error = "CenterPoint Detect returned false";
      }
    } catch (const std::exception& e) {
      result.success = false;
      result.error = std::string("CenterPoint exception: ") + e.what();
    } catch (...) {
      result.success = false;
      result.error = "Unknown CenterPoint exception";
    }

    const auto completed_at = std::chrono::steady_clock::now();
    const auto queue_ms = std::chrono::duration<double, std::milli>(
        started_at - request.enqueued_at).count();
    const auto service_ms = std::chrono::duration<double, std::milli>(
        completed_at - started_at).count();

    result.timing.load_ms = queue_ms;

    if (balancer_) {
      balancer_->complete(worker_index_, result.success, service_ms);
    }

    if (!result.success) {
      std::cerr << "Worker NPU" << device_id_ << " frame " << result.frame_id
                << " failed: " << result.error << "\n";
    }

    if (result_callback_) {
      result_callback_(result);
    }
  }
}

// ============================================================================
// CenterPointWorkerPool implementation
// ============================================================================

CenterPointWorkerPool::CenterPointWorkerPool(const Config& config)
    : config_(config) {}

CenterPointWorkerPool::~CenterPointWorkerPool() {
  workers_.clear();
  load_balancer_.reset();
}

bool CenterPointWorkerPool::initialize(std::string* error) {
  if (config_.device_ids.empty()) {
    if (error) *error = "device_ids cannot be empty";
    return false;
  }

  if (config_.merged_model.empty() &&
      (config_.pfe_model.empty() || config_.rpn_model.empty())) {
    if (error) *error = "merged_model or (pfe_model and rpn_model) must be specified";
    return false;
  }

  // Create load balancer
  load_balancer_ = std::make_unique<LoadBalancer>(
      config_.device_ids.size(), config_.strategy);

  // Create and initialize workers
  std::size_t ready_count = 0;
  for (std::size_t i = 0; i < config_.device_ids.size(); ++i) {
    const int device_id = config_.device_ids[i];

    auto worker = std::make_unique<Worker>(
        device_id, config_.pfe_model, config_.rpn_model, config_.merged_model);

    std::string worker_error;
    const bool worker_ready =
        worker->initialize(&worker_error, config_.score_threshold);

    load_balancer_->set_healthy(i, worker_ready);

    if (worker_ready) {
      ++ready_count;
      std::cout << "CenterPoint worker initialized on NPU core " << device_id
                << "\n";

      // Set per-worker result callback that forwards to pool callback
      worker->set_result_callback([this](const DetectionResult& result) {
        if (result_callback_) {
          result_callback_(result);
        }
      });

      worker->start(i, load_balancer_.get());
    } else {
      std::cerr << "Failed to initialize NPU core " << device_id << ": "
                << worker_error << "\n";
    }

    workers_.push_back(std::move(worker));
  }

  if (ready_count == 0) {
    if (error) *error = "No workers successfully initialized";
    return false;
  }

  initialized_ = true;
  std::cout << "CenterPointWorkerPool initialized with " << ready_count
            << "/" << config_.device_ids.size() << " workers\n";
  return true;
}

bool CenterPointWorkerPool::submit(DetectionRequest request) {
  if (!initialized_) {
    return false;
  }

  // Acquire a worker from load balancer
  auto selected = load_balancer_->acquire();
  if (!selected.has_value()) {
    std::cerr << "No healthy workers available\n";
    return false;
  }

  const std::size_t worker_index = *selected;
  auto& worker = workers_[worker_index];

  request.frame_id = frame_counter_.fetch_add(1) + 1;
  request.enqueued_at = std::chrono::steady_clock::now();

  if (!worker->enqueue(std::move(request), config_.max_queue_size)) {
    load_balancer_->cancel(worker_index);
    return false;
  }

  return true;
}

void CenterPointWorkerPool::set_result_callback(
    std::function<void(const DetectionResult&)> callback) {
  result_callback_ = std::move(callback);
}

std::vector<WorkerStats> CenterPointWorkerPool::get_stats() const {
  if (!load_balancer_) {
    return {};
  }
  return load_balancer_->snapshot();
}

}  // namespace centerpoint
