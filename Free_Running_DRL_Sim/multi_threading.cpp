//
// Created by chi on 5/15/22.
//

#include "multi_threading.h"
#include <iostream>

MultiThreading::MultiThreading(std::shared_ptr<std::atomic<bool>> finish) : finish(std::move(finish)) {

}

void MultiThreading::start_threads(int num_threads) {
    for (int i = 0; i < num_threads; i++) {
        duration.push_back(0);
        num_iterations.push_back(0);
        m_threads.emplace_back([this, i] { this->main_loop(i); });
    }
}

void MultiThreading::join_threads() {
    for (auto &thread: m_threads) {
        thread.join();
    }
}

void MultiThreading::main_loop(int thread_idx) {
    // record the time and the number of iterations
    auto start = high_resolution_clock::now();
    while (true) {
        this->run_one_iteration();
        num_iterations.at(thread_idx) += 1;
        if (*finish) {
            std::cout << "Thread " << this->m_name << " finishes" << std::endl;
            this->post_finish();
            break;
        }
    }
    auto stop = high_resolution_clock::now();
    duration.at(thread_idx) = ((double) duration_cast<milliseconds>(stop - start).count()) / 1000.;
}

std::map<std::string, std::vector<double>> MultiThreading::get_stats() {
    return {
            {"duration (s)", duration},
            {"iterations",   num_iterations}
    };
}

std::string MultiThreading::get_status() const {
    std::lock_guard<std::mutex> lock(m);
    return status;
}

void MultiThreading::update_status(const std::string &s) {
    std::lock_guard<std::mutex> lock(m);
    this->status = s;
}

void MultiThreading::set_name(const std::string &name) {
    this->m_name = name;
}

void MultiThreading::post_finish() {

}
