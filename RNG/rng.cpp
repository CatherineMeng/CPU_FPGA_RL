#include "./rng.h"

extern "C"{
// void pseudo_random(unsigned int seed, int load, rng_type ret, ap_uint<32> lfsr_transfer) {
void pseudo_random(unsigned int seed, int load, rng_type ret[], int indrng) {
  static ap_uint<32> lfsr;
  // ap_uint<32> lfsr=lfsr_transfer;
  if (load ==1 )
    lfsr = seed;
  bool b_32 = lfsr.get_bit(32-32);
  bool b_22 = lfsr.get_bit(32-22);
  bool b_2 = lfsr.get_bit(32-2);
  bool b_1 = lfsr.get_bit(32-1);
  bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
  lfsr = lfsr >> 1;
  lfsr.set_bit(31, new_bit);

  // u u1;
  // u1.funit=lfsr;
  // ret[indrng]=lfsr;
  rng_type *p = reinterpret_cast<rng_type * >(&lfsr);
  ret[indrng] = *p;

  #ifndef __SYNTHESIS__
    printf("\nLSFR sequence: %s ",lfsr.to_string().c_str());  
    printf("\nlfsr to float: %.12f ",float(lfsr)); 
    printf("\nFixed pt number: %.12f ",float(ret[indrng]));  
  #endif
  // return lfsr.to_uint();
  // return ret;
}

// testbench/simulation:
void rng_batch(unsigned int seed,int load,rng_type out[]){
  rng_type ret[N_learner];
  // ap_uint<32> lfsr;
  pseudo_random(seed, load, ret,0);
  for (int i=1; i<N_learner; i++){
    // pseudo_random(seed, load, ret[i], lfsr);
    pseudo_random(seed, 0, ret,i);
    out[i]=ret[i];
    #ifndef __SYNTHESIS__
    printf ("\nrand = %f, ", float(out[i]));
    #endif
  }
}

void top(unsigned int seed, rng_type out[]){
  rng_type out_local[N_learner];
  rng_batch(seed,1,out_local);
  for (int i=0; i<N_learner; i++){
    out[i]=out_local[i];
  }
}

}