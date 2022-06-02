//
// Created by chi on 5/14/22.
//

#include "replay_manager.h"


#include <utility>

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
// #include "cl_ext_xilinx.h"
#include <cstdio>
#include <cstring>

#include "hls_stream.h"
#include "ap_fixed.h"
// #include "hls_math.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iomanip>

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

        // ------------------------------------------------------------------------------------
        // FPGA: Initialize the OpenCL environment
        // ------------------------------------------------------------------------------------
        cl_int err;
        std::string binaryFile = "top.xclbin";
        unsigned fileBufSize;
        std::vector<cl::Device> devices = get_xilinx_devices();
        devices.resize(1);
        cl::Device device = devices[0];
        this->context = cl::Context(device, NULL, NULL, NULL, &err);
        char *fileBuf = read_binary_file(binaryFile, fileBufSize);
        cl::Program::Binaries bins{{fileBuf, fileBufSize}};
        cl::Program program(this->context, devices, bins, NULL, &err);
        this->q  = cl::CommandQueue(this->context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        //cl::CommandQueue q(this->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // cl::Kernel krnl_k2r(program, "top", &err); //comment out===========
        // cl::Kernel top_tree(program, "Top_tree", &err);
        // krnl_init = cl::Kernel(program, "initQ", &err)
        this->krnl_init = cl::Kernel(program, "initQ", &err);
        this->krnl_read1 = cl::Kernel(program, "readQ", &err);
        this->krnl_read2 = cl::Kernel(program, "readQ", &err);
        this->krnl_write = cl::Kernel(program, "writeQ", &err);

        // num_iterations.push_back(0);
        // m_threads.emplace_back([this, 0] { this->main_loop(0); });
        for (int i = 0; i < 1; i++) {
            duration.push_back(0);
            num_iterations.push_back(0);
            m_threads.emplace_back([this, i] { this->main_loop(i); });
        }

        int N_actor=128; //???????????????????Adjust according to tensor size
        // ------------------------------------------------------------------------------------
        // FPGA: Create buffers and initialize test values
        // ------------------------------------------------------------------------------------
        // cl_int err;
        // Create the buffers and allocate memory
        cl::Buffer inpn_buf_act(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, sizeof(float) * N_actor, NULL, &err);
        cl::Buffer inpn_buf_learn(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, sizeof(float) * batch_size, NULL, &err);
        cl::Buffer out_buf(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, sizeof(int) * batch_size, NULL, &err);

        // Map host-side buffer memory to user-space pointers
        float* pn_in_act = (float*) this->q.enqueueMapBuffer(inpn_buf_act, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * N_actor);
        float* pn_in_learn = (float*) this->q.enqueueMapBuffer(inpn_buf_learn, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * batch_size);
        float* ind_o_out = (float*) this->q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, sizeof(float) * batch_size);

        int k;
        for (k = 0; k < N_actor; k++) {
            pn_in_act[k] = 0; //0 is for init on-chip content. At runtime (mainloop), should be pn_in_act[i]=queue.pop()...
        }
        for (k = 0; k < batch_size; k++) {
            pn_in_learn[k]=0;
            ind_o_out[k]=0;
        }

        printf("setArg for initializing\n");

        int i_init=1;

        this->krnl_init.setArg(0, i_init);
        this->krnl_read1.setArg(0, inpn_buf_act);
        this->krnl_read1.setArg(2, N_actor);
        this->krnl_read2.setArg(0, inpn_buf_learn);
        this->krnl_read2.setArg(2, batch_size);
        this->krnl_write.setArg(1, out_buf);
        this->krnl_write.setArg(2, batch_size);

        printf("setArg finished\n");
        // ------------------------------------------------------------------------------------
        // FPGA: Run the kernel for init
        // ------------------------------------------------------------------------------------

       // Copy input data to device global memory
        // std::cout << "Copying data..." << std::endl;
        this->q.enqueueMigrateMemObjects({inpn_buf_act}, 0 /*0 means from host*/);
        this->q.enqueueMigrateMemObjects({inpn_buf_learn}, 0 /*0 means from host*/);

        this->q.finish();

        // Launch the Kernel
        // std::cout << "Launching FPGA queues..." << std::endl;
        this->q.enqueueTask(this->krnl_init);
        this->q.enqueueTask(this->krnl_read1);
        this->q.enqueueTask(this->krnl_read2);
        this->q.enqueueTask(this->krnl_write);

        // wait for all kernels to finish their operations
        this->q.finish();

        // Copy Result from Device Global Memory to Host Local Memory
        this->q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        this->q.finish();

        printf("Init. finished\n");

}

torch::Tensor ReplayManager::generate_idx() const {
    return torch::randint(capacity, {batch_size}, torch::TensorOptions().dtype(torch::kInt64));
}

void ReplayManager::update_priority(const torch::Tensor &priority) const {

}

void ReplayManager::insert_priority(const torch::Tensor &priority) const {

}

// void ReplayManager::start_fpga_host_thread(int N_actor, int batch_size, cl::CommandQueue &q, cl::Kernel &krnl_init, cl::Kernel &krnl_read1, cl::Kernel &krnl_read2, cl::Kernel &krnl_write, cl::this->context this->context) {

// }

void ReplayManager::run_one_iteration() {

    // how to schedule between sampling, priority insertion and priority update
    // auto index = generate_idx();

    this->update_status("Try dequeue from new priority queue");
    auto data_ins = this->new_priority_queue->try_dequeue();
    // update priority
    // if (data.has_value()) {
    //     this->update_priority(data.value()["priorities"]);
    // }
    // insert priority
    this->update_status("Try dequeue from init priority queue");
    auto data_upd = this->init_priority_queue->try_dequeue();
    // if (data.has_value()) {
    //     this->insert_priority(data.value()["init_priorities"]);
    // }


    cl_int err;
    // Create the buffers and allocate memory

    int N_actor=data_ins.value()["init_priorities"].numel();
    cl::Buffer inpn_buf_act(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, sizeof(float) * N_actor, NULL, &err);
    cl::Buffer inpn_buf_learn(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, sizeof(float) * batch_size, NULL, &err);
    cl::Buffer out_buf(this->context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY, sizeof(int) * batch_size, NULL, &err);

    // Map host-side buffer memory to user-space pointers
    float* pn_in_act = (float*)this->q.enqueueMapBuffer(inpn_buf_act, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * N_actor);
    float* pn_in_learn = (float* )this->q.enqueueMapBuffer(inpn_buf_learn, CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * batch_size);
    float* ind_o_out = (float* )this->q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_WRITE | CL_MAP_READ, 0, sizeof(float) * batch_size);

    

    int k;

    // for (k = 0; k < N_actor; k++) {
    //     pn_in_act[k] = data_ins.value()["init_priorities"].index({k}); //0 is for init. At runtime (mainloop), should be pn_in_act[i]=queue.pop()...
    // }
    for (k = 0; k < data_ins.value()["init_priorities"].numel(); ++k)
    {
      pn_in_act[k] = data_ins.value()["init_priorities"].index({k}).item().to<float>();
    }
    for (k = 0; k < batch_size; k++) {
        pn_in_learn[k]=data_upd.value()["priorities"].index({k}).item().to<float>();
        ind_o_out[k]=0;
    }

    printf("setArg for initializing\n");

    int i_init=1;

    this->krnl_init.setArg(0, i_init);
    this->krnl_read1.setArg(0, inpn_buf_act);
    this->krnl_read1.setArg(2, N_actor);
    this->krnl_read2.setArg(0, inpn_buf_learn);
    this->krnl_read2.setArg(2, batch_size);
    this->krnl_write.setArg(1, out_buf);
    this->krnl_write.setArg(2, batch_size);

    printf("setArg finished\n");
    // ------------------------------------------------------------------------------------
    // FPGA: Run the kernel
    // ------------------------------------------------------------------------------------

   // Copy input data to device global memory
    // std::cout << "Copying data..." << std::endl;
    this->q.enqueueMigrateMemObjects({inpn_buf_act}, 0 );
    this->q.enqueueMigrateMemObjects({inpn_buf_learn}, 0 );

    // q.finish();

    // Launch the Kernel
    // std::cout << "Launching FPGA queues..." << std::endl;
    this->q.enqueueTask(this->krnl_init);
    this->q.enqueueTask(this->krnl_read1);
    this->q.enqueueTask(this->krnl_read2);
    this->q.enqueueTask(this->krnl_write);

    // wait for all kernels to finish their operations
    // q.finish();

    // Copy Result from Device Global Memory to Host Local Memory
    this->q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    this->q.finish();

    auto thearray = torch::zeros(batch_size,torch::kFloat64); //or use kF64
    for (k = 0; k < batch_size; k++) {
       // (torch::kFloat64) 
        thearray.index_put_({k},ind_o_out[k]);
    }
    // this->update_status("Enqueue to new priority queue");
    this->update_status("Enqueue storage request queue");
    // this->storage_request_queue->try_enqueue({{"index", index}});
    std::map<std::string, torch::Tensor> m { {"index", thearray}};
    this->storage_request_queue->try_enqueue(m);

}
