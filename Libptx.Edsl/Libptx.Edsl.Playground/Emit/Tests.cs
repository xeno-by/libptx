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
            // add various overloads for this ctor (same as for AtomAttribute)
            var module = new Module(SoftwareIsa.PTX_14, HardwareIsa.SM_13);

            // when vars are created, align gets defaulted to sizeof(type), space will get defaulted to param when added to Params collection
            // add appropriate ctors for Func and Entry, introduce mirrored AddFunc methods
            //
            // also add the Signature method that provides bulk-set of the signature outside of the ctor:
            // 1) native lambda form: (uint a_width, uint a_height, uint a_raw, uint b_width, uint b_height, uint b_raw, uint c_width, uint c_height, uint c_raw) => {}
            // 2) esoteric lambda form: a => b8[12].align4, b => b8[12].align4, c => b8[12].align4
            //
            // what about return values?
            // 1) multiple return values needs to be specified manually
            // 2) native lambda form might specify them by providing non-empty body, e.g. "(int foo) => default(float)"
            // 3) esoteric lambda form might specify them by appending "_ret" to parameter names
            //
            // ahh... why don't we have static import?!
            // all those shortcuts like "b32" can be imported by implementing an interface and wiring those with static helper
            // also we could provide a regioned copy/paste with default implementation of those symbols
            var kernel = new Entry("MatMulKernel", a => b8[12].align4, b => b8[12].align4, c => b8[12].align4);

            // __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
            var a = kernel.Params[0], b = kernel.Params[1], c = kernel.Params[2];
            Label loop_body, after_loop, exit;
            Var_S32 a_width, a_height, a_raw, b_width, b_height, b_raw, row, col, cvalue, dim;
            Var_S32 a_offset, a_offset_lo, a_offset_stride, a_offset_hi;
            Var_S32 b_offset, b_offset_lo, b_offset_stride, b_offset_hi;
            kernel.def(out loop_body, out after_loop, out exit) // out varargs? "def", but not "def_label" since here we can facilitate ad-hoc overloading
            .def(out a_width, out a_height, out a_raw, out b_width, out b_height, out b_raw, out row, out col, out cvalue, out dim)
            .def(out a_offset, out a_offset_lo, out a_offset_stride, out a_offset_hi)
            .def(out b_offset, out b_offset_lo, out b_offset_stride, out b_offset_hi)

            // int row = blockIdx.y * blockDim.y + threadIdx.y;
            // int col = blockIdx.x * blockDim.x + threadIdx.x;
           .mov(rh1, ctaid.x) // full form is rh[1], predefined registers: rh<100>, r<100>, rl<100>, f<100>, fd<100>, p<100> + their unsigned versions
           .mov(rh2, ntid.x)
           .mul.wide(r1, rh1, rh2)
           .mov(rh3, ctaid.y)
           .mov(rh4, ntid.y)
           .mul.wide(r2, rh3, rh4)
           .cvt(r3, tid.x) // exact types since they determine the signature
           .add(col, r3, r1)
           .cvt(r5, tid.y)
           .add(row, r5, r2)

            // if (A.height <= row || B.width <= col) return;
           .ld.param(b_width, b + 0)
           .ld.param(a_height, a + 4)
           .set.le(p6, a_height.u32, row.u32) // notice the ".u32" qualifier
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
