#include <ATen/ATen.h>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void bboxOverlapsKernel(scalar_t *bboxes1, scalar_t *bboxes2,
                                   scalar_t *ious, int num_bboxes1,
                                   int num_bboxes2, int size_bboxes,
                                   int output_size, int mode, bool isAligned) {
  CUDA_1D_KERNEL_LOOP(index, output_size) {
    int b1, b2;
    if (isAligned)
      b1 = index % num_bboxes1, b2 = index % num_bboxes2;
    else
      b1 = index / num_bboxes2, b2 = index % num_bboxes2;
    int offset_b1 = b1 * size_bboxes;
    int offset_b2 = b2 * size_bboxes;

    scalar_t b1_x1 = bboxes1[offset_b1];
    scalar_t b1_y1 = bboxes1[offset_b1 + 1];
    scalar_t b1_x2 = bboxes1[offset_b1 + 2];
    scalar_t b1_y2 = bboxes1[offset_b1 + 3];

    scalar_t b2_x1 = bboxes2[offset_b2];
    scalar_t b2_y1 = bboxes2[offset_b2 + 1];
    scalar_t b2_x2 = bboxes2[offset_b2 + 2];
    scalar_t b2_y2 = bboxes2[offset_b2 + 3];

    scalar_t area1 =
        (b1_x2 - b1_x1 + (scalar_t)1.0f) * (b1_y2 - b1_y1 + (scalar_t)1.0f);

    scalar_t left = fmax(b1_x1, b2_x1);
    scalar_t top = fmax(b1_y1, b2_y1);
    scalar_t right = fmin(b1_x2, b2_x2);
    scalar_t bottom = fmin(b1_y2, b2_y2);
    scalar_t w = fmax(right - left + (scalar_t)1.0f, (scalar_t)0.0f);
    scalar_t h = fmax(bottom - top + (scalar_t)1.0f, (scalar_t)0.0f);
    scalar_t overlap = w * h;

    if (mode == 0) {
      scalar_t area2 =
          (b2_x2 - b2_x1 + (scalar_t)1.0f) * (b2_y2 - b2_y1 + (scalar_t)1.0f);
      ious[index] = overlap / (area1 + area2 - overlap);
    } else
      ious[index] = overlap / area1;
  }
}

void bboxOverlapsLauncher(at::Tensor bboxes1, int num_bboxes1,
                          at::Tensor bboxes2, int num_bboxes2, at::Tensor ious,
                          int out_size, int mode, bool isAligned) {
  int size_bboxes = bboxes1.size(-1);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      bboxes1.scalar_type(), "bboxOverlaps", ([&] {
        scalar_t *bboxes1_data = bboxes1.data<scalar_t>();
        scalar_t *bboxes2_data = bboxes2.data<scalar_t>();
        scalar_t *ious_data = ious.data<scalar_t>();

        bboxOverlapsKernel<scalar_t>
            <<<GET_BLOCKS(out_size), THREADS_PER_BLOCK>>>(
                bboxes1_data, bboxes2_data, ious_data, num_bboxes1, num_bboxes2,
                size_bboxes, out_size, mode, isAligned);
      }));
}