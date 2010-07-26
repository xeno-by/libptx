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
            var a = param.align4.b8[12], b = param.align4.b8[12], c = param.align4.b8[12];
            var kernel = new Entry("MatMulKernel", a, b, c);
            kernel.Params.SetNames("A", "B", "C");

            var loop_body = label, after_loop = label, exit = label;
            var a_width = reg.u32, a_height = reg.u32, a_raw = reg.u32, b_width = reg.u32, b_height = reg.u32, b_raw = reg.u32, c_width = reg.u32, c_height = reg.u32, c_raw = reg.u32;
            var row = reg.u32, col = reg.u32, cvalue = reg.f32, dim = reg.u32;
            var a_offset = reg.u32, a_offset_lo = reg.u32, a_offset_stride = reg.u32, a_offset_hi = reg.u32;
            var b_offset = reg.u32, b_offset_lo = reg.u32, b_offset_stride = reg.u32, b_offset_hi = reg.u32;

            // int row = blockIdx.y * blockDim.y + threadIdx.y;
            // int col = blockIdx.x * blockDim.x + threadIdx.x;
           .mov.u16(rh1, ctaid.x) // full form is rh[1], predefined registers: rh<100>, r<100>, rl<100>, f<100>, fd<100>, p<100> + their unsigned versions
           .mov.u16(rh2, ntid.x)
           .mul.wide.u16(r1, rh1, rh2)
           .mov.u16(rh3, ctaid.y)
           .mov.u16(rh4, ntid.y)
           .mul.wide.u16(r2, rh3, rh4)
           .cvt.u32.u16(r3, tid.x)
           .add.u32(col, r3, r1)
           .cvt.u32.u16(r5, tid.y)
           .add.u32(row, r5, r2)

            // if (A.height <= row || B.width <= col) return;
           .ld.param.u32(b_width, b + 0)
           .ld.param.u32(a_height, a + 4)
           .setp.le.u32(p6, a_height, row)
           .setp.le.u32(p7, b_width, col)
           .or.pred(p1, p6, p7)
           .@(p1).bra(exit)

           // float Cvalue = 0;
           .mov.f32(cvalue, 0)

           // for (int dim = 0; dim < A.width; ++dim)
           .ld.param(a_width, a + 0)
           .mov(dim, 0)
           .setp.le(p2|p8, a_width, dim)
           .@(!p8).bra(after_loop)

           // Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col];
           .ld.param.u32(a_raw, a + 8)
           .mul.lo.u32(r18, a_width, row)
           .mul.lo.u32(a_offset_lo, r18, 4)
           .add.u32(a_offset, a_raw, a_offset_lo)
           .add.u32(r21, r18, a_width)
           .mul.lo.u32(r25, r21, 4)
           .add.u32(a_offset_hi, r25, a_raw)
           .ld.param.u32(b_raw, b + 8)
           .mul.lo.u32(b_offset_lo, col, 4)
           .add.u32(b_offset, b_raw, b_offset_lo)
           .mul.lo.u32(b_offset_stride, b_width, 4)

           // Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col];
           .mark(loop_body)
           .ld.global.f32(f2, a_offset)
           .ld.global.f32(f3, b_offset)
           .mad.f32(cvalue, f3, f2, cvalue)
           .add.u32(a_offset, a_offset, 4)
           .add.u32(b_offset, b_offset, b_offset_stride)
           .setp.ne.u32(p3, a_offset, a_offset_hi)
           .@(p3).bra(loop_body)
           .bra_uni(after_loop)

           // C.elements[row * C.width + col] = Cvalue;
           .ld.param.u32(c_raw, c + 8)
           .ld.param.u32(c_width, c + 0)
           .mul.lo.u32(r32, c_width, row)
           .add.u32(r33, col, r32)
           .mul.lo.u32(r34, r33, 4)
           .add.u32(r35, c_raw, r34)
           .st.global.f32(r35, cvalue)
           .exit();
        }
    }
}
