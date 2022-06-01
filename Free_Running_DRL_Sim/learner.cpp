//
// Created by chi on 5/14/22.
//

#include "learner.h"

#include <utility>

Learner::Learner(int training_time_ms, std::shared_ptr<std::atomic<bool>> finish,
                 std::shared_ptr<SafeQueue<Transition>> training_data_queue,
                 std::shared_ptr<SafeQueue<Transition>> new_priority_queue) :
        MultiThreading(std::move(finish)),
        training_time_ms(training_time_ms),
        new_priority_queue(std::move(new_priority_queue)),
        training_data_queue(std::move(training_data_queue)) {

}

Transition Learner::train(const Transition &data) const {
    std::this_thread::sleep_for(std::chrono::milliseconds(training_time_ms));
    // return random new priorities
    int batch_size = (int) data.begin()->second.size(0);

    auto priorities = torch::rand({batch_size}, torch::TensorOptions().dtype(torch::kFloat64));
    return {
            {"priorities", priorities}
    };
}

void Learner::run_one_iteration() {
    // load data for training
    this->update_status("Dequeue from training data queue");
    auto data = training_data_queue->dequeue();
    // train
    this->update_status("Training");
    auto priority = this->train(data);
    // send back the new priorities
    this->update_status("Enqueue to new priority queue");
    new_priority_queue->enqueue(priority);
}
