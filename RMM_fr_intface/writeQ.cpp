#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

extern "C" {
void writeQ(hls::stream<ap_axiu<32, 0, 0, 0> >& stream, int* buf, int size) {
mem_write:
    for (int i = 0; i < size; i++) {
        ap_axiu<32, 0, 0, 0> v = stream.read();
        buf[i] = v.data;
    }
}
}