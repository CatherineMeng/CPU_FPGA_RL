//
// Created by chi on 5/14/22.
//

#ifndef TPDS22_CODE_DATA_STORAGE_H
#define TPDS22_CODE_DATA_STORAGE_H


#include "transition.h"
#include "safe_queue.h"
#include "multi_threading.h"

#include <torch/torch.h>

using namespace torch::indexing;

class DataStorage : public MultiThreading {
public:
    explicit DataStorage(int capacity,
                         EnvSpec spec,
                         std::shared_ptr<std::atomic<bool>> finish,
                         std::shared_ptr<SafeQueue<Transition>> storage_request_queue,
                         std::shared_ptr<SafeQueue<Transition>> training_data_queue);

    void run_one_iteration() override;

private:
    std::shared_ptr<SafeQueue<Transition>> storage_request_queue;
    std::shared_ptr<SafeQueue<Transition>> training_data_queue;
    EnvSpec spec;

    // storages of data
    Transition storage;
    int capacity;
    int m_ptr = 0;
    int m_size = 0;

};


#endif //TPDS22_CODE_DATA_STORAGE_H
