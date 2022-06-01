//
// Created by chi on 5/15/22.
//

#ifndef TPDS22_CODE_MULTI_THREADING_H
#define TPDS22_CODE_MULTI_THREADING_H

#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>

using namespace std::chrono;

class MultiThreading {
public:
    explicit MultiThreading(std::shared_ptr<std::atomic<bool>> finish);

    virtual void run_one_iteration() = 0;

    void main_loop(int thread_idx);

    void start_threads(int num_threads);

    void join_threads();

    std::map<std::string, std::vector<double>> get_stats();

    void update_status(const std::string &s);

    std::string get_status() const;

    void set_name(const std::string &name);

    virtual void post_finish();

protected:
    std::vector<std::thread> m_threads;
    std::shared_ptr<std::atomic<bool>> finish;
    // statistics
    std::vector<double> num_iterations;
    std::vector<double> duration;
    std::string status = "init";
    std::string m_name;
    mutable std::mutex m;
};


#endif //TPDS22_CODE_MULTI_THREADING_H
