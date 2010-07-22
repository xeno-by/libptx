using NUnit.Framework;
using XenoGears.Playground.Framework;

namespace Libptx.Edsl.Playground.Emit
{
    [TestFixture]
    public class Tests : BaseTests
    {
        [Test]
        public void MatMul()
        {
            // __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
            var module = new Ptx14.Sm13.Module();
            var a = b8[12].align4, b = b8[12].align4, c = b8[12].align4;
            var kernel = new Entry("MatMulKernel", a, b, c);
            kernel.Params.SetNames("A", "B", "C");

            var loop_body = label, after_loop = label, exit = label;
            var a_width = s32, a_height = s32, a_raw = s32, b_width = s32, b_height = s32, b_raw = s32, c_width = s32, c_height = s32, c_raw = s32;
            var row = s32, col = s32, cvalue = s32, dim = s32;
            var a_offset = s32, a_offset_lo = s32, a_offset_stride = s32, a_offset_hi = s32;
            var b_offset = s32, b_offset_lo = s32, b_offset_stride = s32, b_offset_hi = s32;

            // int row = blockIdx.y * blockDim.y + threadIdx.y;
            // int col = blockIdx.x * blockDim.x + threadIdx.x;
           .mov(rh1, ctaid.x) // full form is rh[1], predefined registers: rh<100>, r<100>, rl<100>, f<100>, fd<100>, p<100> + their unsigned versions
           .mov(rh2, ntid.x)
           .mul.wide(r1, rh1, rh2)
           .mov(rh3, ctaid.y)
           .mov(rh4, ntid.y)
           .mul.wide(r2, rh3, rh4)
           .cvt.s32.u16(r3, tid.x)
           .add(col, r3, r1)
           .cvt(r5, tid.y)
           .add(row, r5, r2)

            // if (A.height <= row || B.width <= col) return;
           .ld.param(b_width, b + 0)
           .ld.param(a_height, a + 4)
           .set.le(p6, a_height, row)
           .set.le(p7, b_width, col)
           .or(p1, p6, p7)
           .@(p1).bra(exit)

           // float Cvalue = 0;
           .mov(cvalue, 0)

           // for (int dim = 0; dim < A.width; ++dim)
           .ld.param(a_width, a + 0)
           .mov(dim, 0)
           .set.le(p2|p8, a_width, dim)
           .@(!p8).bra(after_loop)

           // Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col];
           .ld.param(a_raw, a + 8)
           .mul.lo(r18, a_width, row)
           .mul.lo(a_offset_lo, r18, 4)
           .add(a_offset, a_raw, a_offset_lo)
           .add(r21, r18, a_width)
           .mul.lo(r25, r21, 4)
           .add(a_offset_hi, r25, a_raw)
           .ld.param(b_raw, b + 8)
           .mul.lo(b_offset_lo, col, 4)
           .add(b_offset, b_raw, b_offset_lo)
           .mul(b_offset_stride, b_width, 4)

           // Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col];
           .mark(loop_body)
           .ld.global(f2, a_offset)
           .ld.global(f3, b_offset)
           .mad(cvalue, f3, f2, cvalue)
           .add(a_offset, a_offset, 4)
           .add(b_offset, b_offset, b_offset_stride)
           .set.ne(p3, a_offset, a_offset_hi)
           .@(p2).bra(loop_body)
           .bra_uni(after_loop)

           // C.elements[row * C.width + col] = Cvalue;
           .ld.param(c_raw, c + 8)
           .ld.param(c_width, c + 0)
           .mul.lo(r32, c_width, row)
           .add(r33, col, r32)
           .mul.lo(r34, r33, 4)
           .add(r35, c_raw, r34)
           .st.global(r35, cvalue)
           .exit();
        }
    }
}
