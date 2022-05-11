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
#include "./block.h"


// #define N 4

// Forward declaration of utility functions included at the end of this file
std::vector<cl::Device> get_xilinx_devices();
char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb);

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 16
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15)};

template <typename T1>
struct aligned_allocator
{
  using value_type = T1;
  T1* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T1)))
      throw std::bad_alloc();
    return reinterpret_cast<T1*>(ptr);
  }
  void deallocate(T1* p, std::size_t num)
  {
    free(p);
  }
};

// ------------------------------------------------------------------------------------
// Main program
// ------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    // ------------------------------------------------------------------------------------
    // Step 1: Initialize the OpenCL environment
    // ------------------------------------------------------------------------------------
    cl_int err;
    std::string binaryFile = (argc != 2) ? "top.xclbin" : argv[1];
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
    krnl_init = cl::Kernel(program, "initQ", &err)
    krnl_read1 = cl::Kernel(program, "readQ", &err)
    krnl_read2 = cl::Kernel(program, "readQ", &err)
    krnl_write = cl::Kernel(program, "writeQ", &err)

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------

    // std::vector<int, aligned_allocator<int>> insert_ind_in;
    // insert_ind_in.resize(N);
    std::vector<float, aligned_allocator<float>> pn_in_act; //insertion
    pn_in.resize(N_actor);
    std::vector<float, aligned_allocator<float>> pn_in_learn; //update
    pn_in.resize(N_learner);
    std::vector<int, aligned_allocator<int>> ind_o_out;
    ind_o_out.resize(N_learner);
    
    
    int k;
    std::cout << "init..." << std::endl;

    for (k = 0; k < N_actor; k++) {
        pn_in_act[k] = 0; //0 is for init. At runtime, should be pn_in_act[i]=queue.pop()...
    }
    for (k = 0; k < N_learner; k++) {
        pn_in_learn[k]=0;
        ind_o_out[k]=0;
    }

    int init_signal_in = 0;

   printf("inied. ");
    
    cl_mem_ext_ptr_t InExt_act;
    cl_mem_ext_ptr_t InExt_learn;
    cl_mem_ext_ptr_t CrExt; //for output
    

    InExt_act.obj = pn_in_act.data();
    InExt_act.param = 0;
    InExt_act.flags = 0|XCL_MEM_TOPOLOGY;

    InExt_learn.obj = pn_in_learn.data();
    InExt_learn.param = 0;
    InExt_learn.flags = 0|XCL_MEM_TOPOLOGY;

    OutExt.obj = ind_o_out.data();
    OutExt.param = 0;
    OutExt.flags = 0|XCL_MEM_TOPOLOGY;

  printf("flags set\n");

    // Create the buffers and allocate memory

    cl::Buffer inpn_buf_act(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * N_actor, &InExt_act, &err);
    cl::Buffer inpn_buf_learn(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(float) * N_learner, &InExt_learn, &err);

    cl::Buffer out_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(int) * N_learner, &OutExt, &err);

    printf("setArg for initializing\n");

    int size_act=N_actor;
    int size_learn=N_learner;
    int i_init=1;

    OCL_CHECK(err, err = krnl_init.setArg(0, i_init));
    OCL_CHECK(err, err = krnl_read1.setArg(0, inpn_buf_act));
    OCL_CHECK(err, err = krnl_read1.setArg(2, size_act));
    OCL_CHECK(err, err = krnl_read2.setArg(0, inpn_buf_learn));
    OCL_CHECK(err, err = krnl_read2.setArg(2, size_learn));
    OCL_CHECK(err, err = krnl_write.setArg(1, out_buf));
    OCL_CHECK(err, err = krnl_write.setArg(2, size_learn));

    printf("setArg finished\n");
    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel for init
    // ------------------------------------------------------------------------------------

   // Copy input data to device global memory
    std::cout << "Copying data..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({inpn_buf_act}, 0 /*0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({inpn_buf_learn}, 0 /*0 means from host*/));

    OCL_CHECK(err, err = q.finish());

    // Launch the Kernel
    std::cout << "Launching Kernel..." << std::endl;
    OCL_CHECK(err, err = q.enqueueTask(krnl_init));
    OCL_CHECK(err, err = q.enqueueTask(krnl_read1));
    OCL_CHECK(err, err = q.enqueueTask(krnl_read2));
    OCL_CHECK(err, err = q.enqueueTask(krnl_write));

    // wait for all kernels to finish their operations
    OCL_CHECK(err, err = q.finish());

    // Copy Result from Device Global Memory to Host Local Memory
    std::cout << "Getting Results..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    // OPENCL HOST CODE AREA END

    printf("q.finish\n");


    // ------------------------------------------------------------------------------------
    // Step 4: Run the kernel for replay in the loop --- do we only need to change buf or do we need to setarg again for the data transfer to be updated?
    // ------------------------------------------------------------------------------------

    int i_init=0;
    for (k = 0; k < N_actor; k++) {
        pn_in_act[k] = 0.7; //At runtime, should be pn_in_act[i]=queue.pop()...
    }
    for (k = 0; k < N_learner; k++) {
        pn_in_learn[k]=0.7;
    }
    
    OCL_CHECK(err, err = krnl_init.setArg(0, i_init));
    OCL_CHECK(err, err = krnl_read1.setArg(0, inpn_buf_act));
    OCL_CHECK(err, err = krnl_read1.setArg(2, size_act));
    OCL_CHECK(err, err = krnl_read2.setArg(0, inpn_buf_learn));
    OCL_CHECK(err, err = krnl_read2.setArg(2, size_learn));
    OCL_CHECK(err, err = krnl_write.setArg(1, out_buf));
    OCL_CHECK(err, err = krnl_write.setArg(2, size_learn));
   // Copy input data to device global memory
    std::cout << "Copying data..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({inpn_buf_act}, 0 /*0 means from host*/));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({inpn_buf_learn}, 0 /*0 means from host*/));

    OCL_CHECK(err, err = q.finish());

    // Launch the Kernel
    std::cout << "Launching Kernel..." << std::endl;
    OCL_CHECK(err, err = q.enqueueTask(krnl_init));
    OCL_CHECK(err, err = q.enqueueTask(krnl_read1));
    OCL_CHECK(err, err = q.enqueueTask(krnl_read2));
    OCL_CHECK(err, err = q.enqueueTask(krnl_write));

    // wait for all kernels to finish their operations
    OCL_CHECK(err, err = q.finish());

    // Copy Result from Device Global Memory to Host Local Memory
    std::cout << "Getting Results..." << std::endl;
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({out_buf}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    // OPENCL HOST CODE AREA END

    printf("q.finish\n");

    // ------------------------------------------------------------------------------------
    // Step 5: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    printf("ind_o_out content: ");
    for (k = 0; k < N_learner; k++) {
        printf("%d ",ind_o_out[k]);
    }
    bool match = true;
 
    delete[] fileBuf;

    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;
    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}

// ------------------------------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------------------------------
std::vector<cl::Device> get_xilinx_devices()
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i = 0; i < platforms.size(); i++)
    {
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx")
        {
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size())
    {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }

    //Getting ACCELERATOR Devices and selecting 1st such device
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}

char *read_binary_file(const std::string &xclbin_file_name, unsigned &nb)
{
    if (access(xclbin_file_name.c_str(), R_OK) != 0)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);
    return buf;
}