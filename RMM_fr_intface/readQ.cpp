#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

extern "C" {
void readQ(int* host_buf, hls::stream<ap_axiu<32, 0, 0, 0> >& stream, int size) {
readQ:
    for (int i = 0; i < size; i++) {
        int a = host_buf[i];
        ap_axiu<32, 0, 0, 0> v;
        v.data = a;
        stream.write(v);
    }
}
}