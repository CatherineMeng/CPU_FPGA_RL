//
// Created by chi on 5/14/22.
//

#include "data_storage.h"

#include <utility>

DataStorage::DataStorage(int capacity, EnvSpec spec, std::shared_ptr<std::atomic<bool>> finish,
                         std::shared_ptr<SafeQueue<Transition>> storage_request_queue,
                         std::shared_ptr<SafeQueue<Transition>> training_data_queue) :
        MultiThreading(std::move(finish)),
        storage_request_queue(std::move(storage_request_queue)),
        training_data_queue(std::move(training_data_queue)),
        spec(std::move(spec)),
        capacity(capacity) {
    // combine shape first
    std::vector<int64_t> batch_obs_shape{capacity};
    batch_obs_shape.insert(batch_obs_shape.end(), this->spec.obs_shape.begin(), this->spec.obs_shape.end());
    std::vector<int64_t> batch_act_shape{capacity};
    batch_act_shape.insert(batch_act_shape.end(), this->spec.act_shape.begin(), this->spec.act_shape.end());

    // initialize
    this->storage["obs"] = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(this->spec.obs_dtype));
    if (this->spec.act_dtype == torch::kInt64) {
        this->storage["act"] = torch::randint((int64_t) this->spec.max_act, batch_obs_shape);
    } else {
        this->storage["act"] = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(this->spec.act_dtype));
    }
    this->storage["next_obs"] = torch::rand(batch_obs_shape, torch::TensorOptions().dtype(this->spec.obs_dtype));
    this->storage["reward"] = torch::rand({capacity}, torch::TensorOptions().dtype(torch::kFloat64));
    this->storage["done"] = torch::randint(1, {capacity}, torch::TensorOptions().dtype(torch::kBool));
}

void DataStorage::run_one_iteration() {
    this->update_status("Dequeue from storage request queue");
    auto data = storage_request_queue->dequeue();
    // if it is index, then retrieve the data for training. Otherwise, add to the storage
    if (data.contains("index")) {
        // retrieve the data and add to the training_data_queue for training
        Transition output;
        auto idx = data.at("index");
        for (auto &it: storage) {
            output[it.first] = it.second.index({idx});
        }
        this->update_status("Enqueue training data queue");
        training_data_queue->enqueue(output);

    } else {
        // insert data into storage
        int batch_size = (int) data.begin()->second.size(0);
        for (auto &it: data) {
            AT_ASSERT(batch_size == it.second.sizes()[0]);
            if (m_ptr + batch_size > this->capacity) {
                storage[it.first].index_put_({Slice(m_ptr, None)},
                                             it.second.index({Slice(None, capacity - m_ptr)}));
                storage[it.first].index_put_({Slice(None, batch_size - (capacity - m_ptr))},
                                             it.second.index({Slice(capacity - m_ptr, None)}));
            } else {
                storage[it.first].index_put_({Slice(m_ptr, m_ptr + batch_size)}, it.second);
            }
        }
        m_ptr = (m_ptr + batch_size) % capacity;
        m_size = std::min(m_size + batch_size, capacity);
    }
}

