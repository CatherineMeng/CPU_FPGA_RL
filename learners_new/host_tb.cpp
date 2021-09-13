#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#define In_dim 8 //L1
#define W1_out_dim 8 //L1
#define W2_out_dim 64 //L2
#define action_space_dim 4 //L3

#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <CL/cl2.hpp>
#include "./block_new.h"

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
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Kernel krnl_top(program, "learners_top", &err);

    // ------------------------------------------------------------------------------------
    // Step 2: Create buffers and initialize test values
    // ------------------------------------------------------------------------------------
    //In_dim=BATCH SIZE
    // learners_top(blockvec *S, blockvec *Snt, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],int wsync)
    std::vector<blockvec, aligned_allocator<blockvec>> In_rows;
    In_rows.resize(In_dim*BATCHS);
    // std::vector<blockvec> In_rows_snt(In_dim);
    std::vector<blockvec, aligned_allocator<blockvec>> In_rows_snt;
    In_rows_snt.resize(In_dim*BATCHS);
    // std::vector<w1blockvec> Out_w1bram(W1_out_dim);
    std::vector<w1blockvec, aligned_allocator<w1blockvec>> Out_w1bram;
    Out_w1bram.resize(W1_out_dim);
    // std::vector<w3blockvec> Out_w2bram(W2_out_dim);
    std::vector<w3blockvec, aligned_allocator<w3blockvec>> Out_w2bram;
    Out_w2bram.resize(W2_out_dim);
    // std::vector<blockvec> C_rows(In_dim);

    std::vector<actvec, aligned_allocator<actvec>> In_actions;
    In_actions.resize(BATCHS);

    std::vector<blockvec, aligned_allocator<blockvec>> In_rewards;
    In_rewards.resize(BATCHS);

    std::vector<bsbit, aligned_allocator<bsbit>> In_dones;
    In_dones.resize(BATCHS);
    printf("here 1\n");
    

    
    int i, j, jj;
    std::cout << "init input states." << std::endl;
    printf("\ninput s content:\n");
    
    for (jj = 0; jj < BATCHS; jj++) {   
        for (j = 0; j < L1; j++) {
            for (i = 0; i < BSIZE; i++) {
            In_rows[L1*jj+j].a[i] = float(-j)/float(4.0);
            In_rows_snt[L1*jj+j].a[i] = float(j)/float(4.0);
            printf("%f ",In_rows[j].a[i]);
            printf("%f ",In_rows_snt[j].a[i]);
            }
        }
    }

    printf("\ninput reward/action/done content:\n");

    for (jj = 0; jj < BATCHS; jj++) {   
        for (i = 0; i < BSIZE; i++) {
        // printf("\njj,i:%d,%d\n",jj,i);
        In_actions[jj].a[i] = int(2);
        In_rewards[jj].a[i] = float(1);
        In_dones[jj].a[i] = int(0);
        std::cout << sizeof(int) * BSIZE * BATCHS << std::endl;
        std::cout << sizeof(float) * BSIZE * BATCHS << std::endl;
        // printf("%f ",In_actions[jj].a[i]);
        }
    }
  printf("inied\n");
    
    cl_mem_ext_ptr_t InrExt;
    cl_mem_ext_ptr_t InrExt2;
    cl_mem_ext_ptr_t InrExt3;
    cl_mem_ext_ptr_t InrExt4;
    cl_mem_ext_ptr_t InrExt5;
    cl_mem_ext_ptr_t OutExt;
    cl_mem_ext_ptr_t OutExt2;
    

    InrExt.obj = In_rows.data();
    InrExt.param = 0;
    InrExt.flags = 0|XCL_MEM_TOPOLOGY;

    InrExt2.obj = In_rows_snt.data();
    InrExt2.param = 0;
    InrExt2.flags = 0|XCL_MEM_TOPOLOGY;

    InrExt3.obj = In_actions.data();
    InrExt3.param = 0;
    InrExt3.flags = 0|XCL_MEM_TOPOLOGY;

    InrExt4.obj = In_rewards.data();
    InrExt4.param = 0;
    InrExt4.flags = 0|XCL_MEM_TOPOLOGY;

    InrExt5.obj = In_dones.data();
    InrExt5.param = 0;
    InrExt5.flags = 0|XCL_MEM_TOPOLOGY;

    OutExt.obj = Out_w1bram.data();
    OutExt.param = 0;
    OutExt.flags = 0|XCL_MEM_TOPOLOGY;

    OutExt2.obj = Out_w2bram.data();
    OutExt2.param = 0;
    OutExt2.flags = 0|XCL_MEM_TOPOLOGY;

    // CrExt.obj = C_rows.data();
    // CrExt.param = 0;
    // CrExt.flags = 1|XCL_MEM_TOPOLOGY;
    

    // actvec acts={2}; //consistent with tb. Moved to Aligned allocator
    // blockvec r={1}; //consistent with tb. Moved to Aligned allocator
    float gamma=0.5;
    float alpha=0.1;

    // sglbit done_tmp=0;
    // sglbit done_tmp=0;
    // bsbit done={0}; //consistent with tb. Moved to Aligned allocator
    // for(int i = 0; i < BSIZE; i++) {
    //     done.a[i]=0;  //BS cols

    // }
    
    int wsync = 0;

  printf("flags set\n");
    // Create the buffers and allocate memory
    cl::Buffer in1_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * In_dim * BATCHS, &InrExt, &err);
    cl::Buffer in2_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * In_dim * BATCHS, &InrExt2, &err);
    cl::Buffer in3_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(actvec) * BATCHS, &InrExt3, &err);
    std::cout << sizeof(actvec) * BATCHS << std::endl;
    cl::Buffer in4_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(blockvec) * BATCHS, &InrExt4, &err);
    cl::Buffer in5_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(bsbit) * BATCHS, &InrExt5, &err);
    cl::Buffer out1_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w1blockvec) * W1_out_dim, &OutExt, &err);
    cl::Buffer out2_buf(context, CL_MEM_USE_HOST_PTR|CL_MEM_WRITE_ONLY|CL_MEM_EXT_PTR_XILINX, sizeof(w3blockvec) * W2_out_dim, &OutExt2, &err);
      printf("hi\n");

      // void learners_top(blockvec *S, blockvec *Snt, actvec *acts,blockvec *r,float gamma, float alpha, bsbit *done, w1blockvec w1bram_out[L1],w3blockvec w2bram_out[L2],int wsync)
    // Set kernel arguments
    krnl_top.setArg(0, in1_buf);
    krnl_top.setArg(1, in2_buf);
    krnl_top.setArg(2, in3_buf);
    krnl_top.setArg(3, in4_buf);
    krnl_top.setArg(4, gamma);
    krnl_top.setArg(5, alpha);
    krnl_top.setArg(6, in5_buf);
    krnl_top.setArg(7, out1_buf);
    krnl_top.setArg(8, out2_buf);
    krnl_top.setArg(9, wsync);

    // Map host-side buffer memory to user-space pointers [replaced, used equeueMapBuffer]
    //blockvec *A = (blockvec *)q.enqueueMapBuffer(in1_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockvec) * In_dim);
    //blockvec *B = (blockvec *)q.enqueueMapBuffer(in2_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockvec) * In_dim);
    //blockmat *C = (blockmat *)q.enqueueMapBuffer(out_buf, CL_TRUE, CL_MAP_WRITE, 0, sizeof(blockmat) * W1_out_dim);
    //std::vector<blockvec> A(In_dim);
    //std::vector<blockvec> B(In_dim);
    //std::vector<blockmat> C(W1_out_dim);
    
    printf("setArg finished\n");


    // ------------------------------------------------------------------------------------
    // Step 3: Run the kernel
    // ------------------------------------------------------------------------------------

    printf("setArg\n");
    // Schedule transfer of inputs to device memory, execution of kernel, and transfer of outputs back to host memory
    q.enqueueMigrateMemObjects({in1_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in2_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in3_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in4_buf}, 0 /* 0 means from host*/);
    q.enqueueMigrateMemObjects({in5_buf}, 0 /* 0 means from host*/);
    q.finish();
    printf("sent data\n");
    q.enqueueTask(krnl_top);
    q.finish();
    printf("enqueue\n");
    q.enqueueMigrateMemObjects({out1_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    q.enqueueMigrateMemObjects({out2_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
    printf("executed kernel\n");
    // printf("data back\n");

    // Wait for all scheduled operations to finish
    q.finish();
    printf("q.finish\n");
    // ------------------------------------------------------------------------------------
    // Step 4: Check Results and Release Allocated Resources
    // ------------------------------------------------------------------------------------
    bool match = true;
    printf("hi\n");
    // FILE *fp;
    // fp=fopen("./Crows.dat","w");
    // printf("hi\n");
    // for (j = 0; j < L3; j++) {
    //     for (i = 0; i < BSIZE; i++) {
    //         fprintf(fp, "%f ", C_rows[j].a[i]);
    //     }
    //     fprintf(fp, "\n");
    // }
    // fclose(fp);
 
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