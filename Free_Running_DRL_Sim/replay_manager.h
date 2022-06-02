//
// Created by chi on 5/14/22.
//

#ifndef TPDS22_CODE_REPLAY_MANAGER_H
#define TPDS22_CODE_REPLAY_MANAGER_H

#include "transition.h"
#include "safe_queue.h"
#include "multi_threading.h"

#include "fpga_rmm.h"

class ReplayManager : public MultiThreading {
public:
    ReplayManager(std::shared_ptr<std::atomic<bool>> finish,
                  std::shared_ptr<SafeQueue<Transition>> storage_request_queue,
                  std::shared_ptr<SafeQueue<Transition>> new_priority_queue,
                  std::shared_ptr<SafeQueue<Transition>> init_priority_queue,
                  int batch_size,
                  int capacity);

    torch::Tensor generate_idx() const;

    void update_priority(const torch::Tensor &priority) const;

    void insert_priority(const torch::Tensor &priority) const;

    // void start_fpga_host_thread(int N_actor, int N_learner, cl::CommandQueue &q, cl::Kernel &krnl_init, cl::Kernel &krnl_read1, cl::Kernel &krnl_read2, cl::Kernel &krnl_write, cl::Context context);

    void run_one_iteration() override;

    // void run_one_iteration_fpga(int N_actor, int N_learner, cl::CommandQueue &q, cl::Kernel &krnl_init, cl::Kernel &krnl_read1, cl::Kernel &krnl_read2, cl::Kernel &krnl_write, cl::Context context);
     // override;


private:
    std::shared_ptr<SafeQueue<Transition>> storage_request_queue;
    std::shared_ptr<SafeQueue<Transition>> new_priority_queue;
    std::shared_ptr<SafeQueue<Transition>> init_priority_queue;
    int batch_size;
    int capacity;

    // int N_actor,
    cl::CommandQueue q;
    cl::Kernel krnl_init;
    cl::Kernel krnl_read1;
    cl::Kernel krnl_read2;
    cl::Kernel krnl_write;
    cl::Context context;
};


#endif //TPDS22_CODE_REPLAY_MANAGER_H
