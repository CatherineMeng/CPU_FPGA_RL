
#include "ap_fixed.h"
#include <iostream>
#include <iomanip>
#include <vector>

extern "C"{
using namespace std;

typedef ap_ufixed<32,0> rng_type;

// typedef union{
// 	ap_uint<32> funit;
// 	rng_type rngt;
// } u;
#define N_learner 128
// void pseudo_random(unsigned int seed, int load, rng_type ret, ap_uint<32> lfsr_transfer);
void pseudo_random(unsigned int seed, int load, rng_type ret[], int indrng);
void rng_batch(unsigned int seed,int load,rng_type out[]);
void top(unsigned int seed, rng_type out[]);

}