//
// Created by chi on 5/15/22.
//

// create a pybinding to test the system in Python for efficient timing


#include "actor.h"
#include "data_storage.h"
#include "learner.h"
#include "replay_manager.h"
// #include "fpga_rmm.h" //for FPGA array init.
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS


#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include <cstdio>
#include <cstring>
// #include <xcl2.hpp>

// =================rmm.h for kernel========================
#include "hls_stream.h"
#include "ap_fixed.h"
// #include "hls_math.h"
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <iomanip>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <torch/torch.h>
#include <torch/python.h>

#include <vector>
#include <map>

namespace py = pybind11;

template<class T>
std::vector<T> convert_py_array_to_vector(const py::array_t<T> &data) {
    std::vector<T> output(data.size());
    for (int i = 0; i < data.size(); i++) {
        output[i] = data.at(i);
    }
    return output;
}

class Simulator {
public:
    Simulator(const py::array_t<int64_t> &obs_shape,
              const py::object &obs_dtype,
              const py::array_t<int64_t> &act_shape,
              const py::object &act_dtype,
              double max_act,
            // data collector parameters
              int num_actors,
              int data_collection_time_ms,
              int local_buffer_size,
            // learner parameters
              int training_time_ms,
            // RMM
              int capacity,
              int batch_size,
              int max_queue_size
    ) {
        EnvSpec spec(convert_py_array_to_vector(obs_shape),
                     torch::python::detail::py_object_to_dtype(obs_dtype),
                     convert_py_array_to_vector(act_shape),
                     torch::python::detail::py_object_to_dtype(act_dtype),
                     max_act);
        finish_signal = std::make_shared<std::atomic<bool>>(false);

        init_priority_queue = std::make_shared<SafeQueue<Transition>>(max_queue_size);
        storage_request_queue = std::make_shared<SafeQueue<Transition>>(max_queue_size);
        training_data_queue = std::make_shared<SafeQueue<Transition>>(max_queue_size);
        new_priority_queue = std::make_shared<SafeQueue<Transition>>(max_queue_size);

        actor = std::make_unique<Actor>(spec, finish_signal, data_collection_time_ms, local_buffer_size,
                                        init_priority_queue, storage_request_queue);
        actor->set_name("actor");
        data_storage = std::make_unique<DataStorage>(capacity, spec, finish_signal, storage_request_queue,
                                                     training_data_queue);
        data_storage->set_name("data_storage");
        learner = std::make_unique<Learner>(training_time_ms, finish_signal,
                                            training_data_queue, new_priority_queue);
        learner->set_name("learner");
        replay_manager = std::make_unique<ReplayManager>(finish_signal, storage_request_queue,
                                                         new_priority_queue, init_priority_queue,
                                                         batch_size, capacity);
        replay_manager->set_name("replay_manager");
        this->num_actors = num_actors;

        /*
        // ------------------------------------------------------------------------------------
        // FPGA: Initialize the OpenCL environment
        // ------------------------------------------------------------------------------------
        cl_int err;
        std::string binaryFile = "top.xclbin";
        unsigned fileBufSize;
        std::vector<cl::Device> devices = get_xilinx_devices();
        devices.resize(1);
        cl::Device device = devices[0];
        cl::Context context(device, NULL, NULL, NULL, &err);
        char *fileBuf = read_binary_file(binaryFile, fileBufSize);
        cl::Program::Binaries bins{{fileBuf, fileBufSize}};
        cl::Program program(context, devices, bins, NULL, &err);
        cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
        //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // cl::Kernel krnl_k2r(program, "top", &err); //comment out===========
        // cl::Kernel top_tree(program, "Top_tree", &err);
        // krnl_init = cl::Kernel(program, "initQ", &err)
        cl::Kernel krnl_init(program, "initQ", &err);
        cl::Kernel krnl_read1(program, "readQ", &err);
        cl::Kernel krnl_read2(program, "readQ", &err);
        cl::Kernel krnl_write(program, "writeQ", &err);
        */

    }

    void start() {
        // start all the threads
        std::cout << "Start threads..." << std::endl;
        actor->start_threads(this->num_actors);
        data_storage->start_threads(1);
        learner->start_threads(1);
        replay_manager->start_threads(1);
    }

    void end() {
        // set finish signal and join the threads
        std::cout << "terminating threads..." << std::endl;
        *finish_signal = true;
        data_storage->join_threads();
        replay_manager->join_threads();
        actor->join_threads();
        learner->join_threads();
    }

    [[nodiscard]] std::map<std::string, std::map<std::string, std::vector<double>>> get_stats() const {
        std::map<std::string, std::map<std::string, std::vector<double>>> results;
        results["actor"] = actor->get_stats();
        results["data_storage"] = data_storage->get_stats();
        results["learner"] = learner->get_stats();
        results["replay_manager"] = replay_manager->get_stats();
        return results;
    }

    [[nodiscard]] std::map<std::string, std::string> get_status() const {
        std::map<std::string, std::string> results;
        results["actor"] = actor->get_status();
        results["data_storage"] = data_storage->get_status();
        results["learner"] = learner->get_status();
        results["replay_manager"] = replay_manager->get_status();
        return results;
    }

    [[nodiscard]] std::map<std::string, int> get_queue_size() const {
        std::map<std::string, int> results;
        results["init_priority_queue"] = init_priority_queue->size();
        results["storage_request_queue"] = storage_request_queue->size();
        results["training_data_queue"] = training_data_queue->size();
        results["new_priority_queue"] = new_priority_queue->size();
        return results;
    }


private:
    std::shared_ptr<std::atomic<bool>> finish_signal;
    // queues
    std::shared_ptr<SafeQueue<Transition>> init_priority_queue;
    std::shared_ptr<SafeQueue<Transition>> storage_request_queue;
    std::shared_ptr<SafeQueue<Transition>> training_data_queue;
    std::shared_ptr<SafeQueue<Transition>> new_priority_queue;
    // objects
    std::unique_ptr<Actor> actor;
    std::unique_ptr<DataStorage> data_storage;
    std::unique_ptr<Learner> learner;
    std::unique_ptr<ReplayManager> replay_manager;
    int num_actors;
};


//PYBIND11_MAKE_OPAQUE(std::vector<double>);

// create binding
PYBIND11_MODULE(tpds, m) {
    py::class_<Simulator>(m, "Simulator")
            .def(py::init<const py::array_t<int64_t> &, const py::object &,
                         const py::array_t<int64_t> &,
                         const py::object &,
                         double,
                         // data collector parameters
                         int,
                         int,
                         int,
                         // learner parameters
                         int,
                         // RMM
                         int,
                         int,
                         int>(),
                 py::arg("obs_shape"),
                 py::arg("obs_dtype"),
                 py::arg("act_shape"),
                 py::arg("act_dtype"),
                 py::arg("max_act"),
                 py::arg("num_actors"),
                 py::arg("data_collection_time_ms"),
                 py::arg("local_buffer_size"),
                 py::arg("training_time_ms"),
                 py::arg("capacity"),
                 py::arg("batch_size"),
                 py::arg("max_queue_size"))
            .def("start", &Simulator::start)
            .def("end", &Simulator::end)
            .def("get_stats", &Simulator::get_stats)
            .def("get_status", &Simulator::get_status)
            .def("get_queue_size", &Simulator::get_queue_size);
}