


#include "utility.h"
using DATATYPE = half;
#include "cutlass/util/device_nchw_to_nhwc.h"
#include "cutlass/util/device_nhwc_to_nchw.h"

void cutlass_nchw_nhwc(const DATATYPE *input, DATATYPE *output, int batch, int ic,
                  int ih, int iw) {
cutlass::Tensor4DCoord a = cutlass::Tensor4DCoord(batch, iw, ic, ih);
cutlass::Tensor4DCoord b = cutlass::Tensor4DCoord(batch, ih, iw, ic);
cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNCHW> c {(cutlass::half_t *)input, cutlass::make_Coord(iw, ih * iw, ic * ih * iw)};
cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> d {(cutlass::half_t *)output, cutlass::make_Coord(iw, iw * ih, ic * iw * ih)};

cutlass::nchw_to_nhwc(a, b, c, d, nullptr);
}



void cutlass_nhwc_nchw(const DATATYPE *input, DATATYPE *output, int batch, int ic,
    int ih, int iw) {

cutlass::Tensor4DCoord a = cutlass::Tensor4DCoord(batch, ih, iw, ic);
cutlass::Tensor4DCoord b = cutlass::Tensor4DCoord(batch, iw, ic, ih);
cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> c {(cutlass::half_t *)input, cutlass::make_Coord(iw, ih * iw, ic * ih * iw)};
cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNCHW> d {(cutlass::half_t *)output, cutlass::make_Coord(iw, iw * ih, ic * iw * ih)};

cutlass::nhwc_to_nchw(a, b, c, d, nullptr);
}

