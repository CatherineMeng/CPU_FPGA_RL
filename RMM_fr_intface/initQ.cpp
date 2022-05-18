#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

extern "C" {
void initQ(int host_sig, hls::stream<ap_axiu<2, 0, 0, 0> >& stream) {
readQ:
    if (host_sig) {
        ap_axiu<2, 0, 0, 0> v;
        v.data = 1;
        stream.write(v);
    }
}
}