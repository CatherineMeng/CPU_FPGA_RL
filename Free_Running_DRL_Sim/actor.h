//
// Created by Chi Zhang on 5/12/22.
//

#ifndef TPDS22_CODE_ACTOR_H
#define TPDS22_CODE_ACTOR_H

#include <thread>
#include <utility>

#include "safe_queue.h"
#include "transition.h"
#include "multi_threading.h"

class Actor : public MultiThreading {
public:
    Actor(EnvSpec spec,
          std::shared_ptr<std::atomic<bool>> finish,
          int data_collection_time_ms,
          int local_buffer_size,
          std::shared_ptr<SafeQueue<Transition>> init_priority_queue,
          std::shared_ptr<SafeQueue<Transition>> storage_request_queue);

    std::pair<Transition, Transition> get_data();

    void run_one_iteration() override;

private:
    std::shared_ptr<SafeQueue<Transition>> init_priority_queue;
    std::shared_ptr<SafeQueue<Transition>> storage_request_queue;
    int data_collection_time_ms;
    EnvSpec spec;
    int local_buffer_size;


};


#endif //TPDS22_CODE_ACTOR_H
