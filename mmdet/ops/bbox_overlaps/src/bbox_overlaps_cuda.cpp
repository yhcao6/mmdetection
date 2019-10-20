#include <ATen/ATen.h>
#include <math.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void bboxOverlapsLauncher(at::Tensor bboxes1, int num_bboxes1,
                          at::Tensor bboxes2, int num_bboxes2, at::Tensor ious,
                          int out_size, int mode, bool isAligned);

at::Tensor bboxOverlaps(at::Tensor bboxes1, at::Tensor bboxes2, at::Tensor ious,
                        int mode = 0, bool isAligned = false) {
  CHECK_CUDA(bboxes1);
  CHECK_CUDA(bboxes2);

  int num_bboxes1 = bboxes1.size(0);
  int num_bboxes2 = bboxes2.size(0);
  int out_size;
  if (isAligned)
    out_size = num_bboxes1;
  else
    out_size = num_bboxes1 * num_bboxes2;
  ious = ious.view(out_size);
  bboxOverlapsLauncher(bboxes1, num_bboxes1, bboxes2, num_bboxes2, ious,
                       out_size, mode, isAligned);
  if (isAligned)
    ious = at::reshape(ious, {num_bboxes1});
  else
    ious = at::reshape(ious, {num_bboxes1, num_bboxes2});
  return ious;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bbox_overlaps", &bboxOverlaps, "bbox overlaps (CUDA)");
}
