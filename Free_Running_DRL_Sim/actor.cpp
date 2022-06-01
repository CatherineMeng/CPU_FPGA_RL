//
// Created by Chi Zhang on 5/12/22.
//

#include "actor.h"

#include <utility>

Actor::Actor(EnvSpec spec, std::shared_ptr<std::atomic<bool>> finish, int data_collection_time_ms,
             int local_buffer_size,
             std::shared_ptr<SafeQueue<Transition>> init_priority_queue,
             std::shared_ptr<SafeQueue<Transition>> storage_request_queue)
        :
        MultiThreading(std::move(finish)),
        init_priority_queue(std::move(init_priority_queue)),
        storage_request_queue(std::move(storage_request_queue)),
        data_collection_time_ms(data_collection_time_ms),
        spec(std::move(spec)),
        local_buffer_size(local_buffer_size) {

}

std::pair<Transition, Transition> Actor::get_data() {
    // run_one_iteration the environment to fill the local buffer and return the data
    // we use sleep + random number to simulate the process
    std::this_thread::sleep_for(std::chrono::milliseconds(data_collection_time_ms));

    std::vector<int64_t> batch_obs_shape{local_buffer_size};
    batch_obs_shape.insert(batch_obs_shape.end(), spec.obs_shape.begin(), spec.obs_shape.end());
    std::vector<int64_t> batch_act_shape{local_buffer_size};
    batch_act_shape.insert(batch_act_shape.end(), spec.act_shape.begin(), spec.act_shape.end());

    // send data to the queue
    auto obs = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(spec.obs_dtype));
    torch::Tensor act;
    if (spec.act_dtype == torch::kInt64) {
        act = torch::randint((int64_t) spec.max_act, batch_obs_shape);
    } else {
        act = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(spec.act_dtype));
    }
    auto next_obs = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(spec.obs_dtype));
    auto reward = torch::rand({local_buffer_size}, torch::TensorOptions().dtype(torch::kFloat64));
    auto done = torch::randint(1, {local_buffer_size}, torch::TensorOptions().dtype(torch::kBool));
    auto init_priority = torch::rand({local_buffer_size}, torch::TensorOptions().dtype(torch::kFloat64));
    return {{
                    {"obs",             obs},
                    {"act", act},
                    {"next_obs", next_obs},
                    {"reward", reward},
                    {"done", done}
            },
            {       {"init_priorities", init_priority}}};
}

void Actor::run_one_iteration() {
    // Actor will never be blocked on waitlist.
    this->update_status("Preparing data");
    auto data = this->get_data();
    this->update_status("Enqueue storage request queue");
    bool success = this->storage_request_queue->try_enqueue(data.first);
    if (success) {
        this->update_status("Enqueue init priority queue");
        this->init_priority_queue->enqueue(data.second);
    } else {
        // yield the thread
        std::this_thread::yield();
    }
}


