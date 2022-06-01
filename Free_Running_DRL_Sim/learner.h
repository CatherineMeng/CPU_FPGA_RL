//
// Created by chi on 5/14/22.
//

#ifndef TPDS22_CODE_LEARNER_H
#define TPDS22_CODE_LEARNER_H

#include <utility>

#include "safe_queue.h"
#include "transition.h"
#include "multi_threading.h"

class Learner : public MultiThreading {
public:
    Learner(int training_time_ms,
            std::shared_ptr<std::atomic<bool>> finish,
            std::shared_ptr<SafeQueue<Transition>> training_data_queue,
            std::shared_ptr<SafeQueue<Transition>> new_priority_queue);

    [[nodiscard]] Transition train(const Transition &data) const;

    void run_one_iteration() override;

    void post_finish() override {
        training_data_queue->try_dequeue();
    }

private:
    int training_time_ms;
    std::shared_ptr<SafeQueue<Transition>> new_priority_queue;
    std::shared_ptr<SafeQueue<Transition>> training_data_queue;

};


#endif //TPDS22_CODE_LEARNER_H
