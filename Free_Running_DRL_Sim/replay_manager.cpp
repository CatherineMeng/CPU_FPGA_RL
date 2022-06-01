//
// Created by chi on 5/14/22.
//

#include "replay_manager.h"

#include <utility>

ReplayManager::ReplayManager(std::shared_ptr<std::atomic<bool>> finish,
                             std::shared_ptr<SafeQueue<Transition>> storage_request_queue,
                             std::shared_ptr<SafeQueue<Transition>> new_priority_queue,
                             std::shared_ptr<SafeQueue<Transition>> init_priority_queue, int batch_size, int capacity)
        : MultiThreading(std::move(finish)),
          storage_request_queue(std::move(storage_request_queue)),
          new_priority_queue(std::move(new_priority_queue)),
          init_priority_queue(std::move(init_priority_queue)),
          batch_size(batch_size),
          capacity(capacity) {

}

torch::Tensor ReplayManager::generate_idx() const {
    return torch::randint(capacity, {batch_size}, torch::TensorOptions().dtype(torch::kInt64));
}

void ReplayManager::update_priority(const torch::Tensor &priority) const {

}

void ReplayManager::insert_priority(const torch::Tensor &priority) const {

}

void ReplayManager::run_one_iteration() {
    // how to schedule between sampling, priority insertion and priority update
    auto index = generate_idx();
    this->update_status("Enqueue storage request queue");
    this->storage_request_queue->try_enqueue({{"index", index}});
    this->update_status("Try dequeue from new priority queue");
    auto data = this->new_priority_queue->try_dequeue();
    // update priority
    if (data.has_value()) {
        this->update_priority(data.value()["priorities"]);
    }
    // insert priority
    this->update_status("Try dequeue from init priority queue");
    data = this->init_priority_queue->try_dequeue();
    if (data.has_value()) {
        this->insert_priority(data.value()["init_priorities"]);
    }
}
