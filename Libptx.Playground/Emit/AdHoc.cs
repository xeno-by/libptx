using System;
using Libcuda.Versions;
using Libptx.Common.Comments;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Expressions.Immediate;
using Libptx.Expressions.Slots;
using Libptx.Expressions.Sregs;
using Libptx.Instructions.Arithmetic;
using Libptx.Instructions.ComparisonAndSelection;
using Libptx.Instructions.ControlFlow;
using Libptx.Instructions.LogicAndShift;
using Libptx.Instructions.MovementAndConversion;
using Libptx.Statements;
using NUnit.Framework;
using XenoGears.Playground.Framework;
using Type=Libptx.Common.Types.Type;
using Libptx.Common;

namespace Libptx.Playground.Emit
{
    [TestFixture]
    public class AdHoc : BaseTests
    {
        [Test, Category("Hot")]
        public void MatMul()
        {
            Func<String, Reg> reg_u32 = name => new Reg{Name = name, Type = new Type{Name = TypeName.U32}};
            Func<String, Reg> reg_f32 = name => new Reg{Name = name, Type = new Type{Name = TypeName.F32}};
            Func<int, Reg> rh = i => new Reg{Name = String.Format("%rh{0}", i), Type = new Type{Name = TypeName.U16}};
            Func<int, Reg> r = i => new Reg{Name = String.Format("%r{0}", i), Type = new Type{Name = TypeName.U32}};
            Func<int, Reg> rd = i => new Reg{Name = String.Format("%rd{0}", i), Type = new Type{Name = TypeName.U64}};
            Func<int, Reg> f = i => new Reg{Name = String.Format("%f{0}", i), Type = new Type{Name = TypeName.F32}};
            Func<int, Reg> fd = i => new Reg{Name = String.Format("%d{0}", i), Type = new Type{Name = TypeName.F64}};
            Func<int, Reg> p = i => new Reg{Name = String.Format("%p{0}", i), Type = new Type{Name = TypeName.Pred}};
            Type u16 = new Type { Name = TypeName.U16 }, s16 = new Type { Name = TypeName.S16 };
            Type u32 = new Type { Name = TypeName.U32 }, s32 = new Type { Name = TypeName.S32 };
            Type f32 = new Type { Name = TypeName.F32 }, f64 = new Type { Name = TypeName.F64 };
            Type pred = new Type { Name = TypeName.Pred };

            var module = new Module(SoftwareIsa.PTX_14, HardwareIsa.SM_13);
            Func<String, Var> param_align4_b8_12 = name => new Var{Name = name, Space = space.param, Alignment = 4, Type = new Type{Name = TypeName.B8, Dims = new []{12}}};
            Var a = param_align4_b8_12("A"), b = param_align4_b8_12("B"), c = param_align4_b8_12("C");
            var kernel = module.AddEntry("MatMulKernel", a, b, c);
            var ptx = kernel.Stmts;

            Func<String, Label> label = name => new Label{Name = name};
            Label loop_body = label("$LoopBody"), after_loop = label("$AfterLoop"), exit = label("$Exit");
            Reg a_width = reg_u32("a_width"), a_height = reg_u32("a_height"), a_raw = reg_u32("a_raw");
            Reg b_width = reg_u32("b_width"), b_height = reg_u32("b_height"), b_raw = reg_u32("b_raw");
            Reg c_width = reg_u32("c_width"), c_height = reg_u32("c_height"), c_raw = reg_u32("c_raw");
            Reg row = reg_u32("row"), col = reg_u32("col"), cvalue = reg_f32("cvalue"), dim = reg_u32("dim");
            Reg a_offset = reg_u32("a_offset"), a_offset_lo = reg_u32("a_offset_lo"), a_offset_stride = reg_u32("a_offset_stride"), a_offset_hi = reg_u32("a_offset_hi");
            Reg b_offset = reg_u32("b_offset"), b_offset_lo = reg_u32("b_offset_lo"), b_offset_stride = reg_u32("b_offset_stride"), b_offset_hi = reg_u32("b_offset_hi");

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "int row = blockIdx.y * blockDim.y + threadIdx.y;"});
            ptx.Add(new Comment{Text = "int col = blockIdx.x * blockDim.x + threadIdx.x;"});
            ptx.Add(new mov{type = u16, d = rh(1), a = new ctaid().mod(Mod.X)});
            ptx.Add(new mov{type = u16, d = rh(2), a = new ntid().mod(Mod.X)});
            ptx.Add(new mul{type = u16, mode = mulm.wide, d = r(1), a = rh(1), b = rh(2)});
            ptx.Add(new mov{type = u16, d = rh(3), a = new ctaid().mod(Mod.Y)});
            ptx.Add(new mov{type = u16, d = rh(4), a = new ntid().mod(Mod.Y)});
            ptx.Add(new mul{type = u16, mode = mulm.wide, d = r(2), a = rh(3), b = rh(4)});
            ptx.Add(new cvt{dtype = u32, atype = u16, d = r(3), a = new tid().mod(Mod.X)});
            ptx.Add(new add{type = u32, d = col, a = r(3), b = r(1)});
            ptx.Add(new cvt{dtype = u32, atype = u16, d = r(5), a = new tid().mod(Mod.Y)});
            ptx.Add(new add{type = u32, d = row, a = r(5), b = r(2)});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "if (A.height <= row || B.width <= col) return;"});
            ptx.Add(new ld{ss = space.param, type = u32, d = b_width, a = b + 0});
            ptx.Add(new ld{ss = space.param, type = u32, d = a_height, a = a + 4});
            ptx.Add(new setp{cmpop = cmp.le, type = u32, p = p(6), a = a_height, b = row});
            ptx.Add(new setp{cmpop = cmp.le, type = u32, p = p(7), a = b_width, b = col});
            ptx.Add(new or{type = pred, d = p(1), a = p(6), b = p(8)});
            ptx.Add(new bra{Guard = p(1), tgt = exit});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "float Cvalue = 0;"});
            ptx.Add(new mov{type = f32, d = cvalue, a = (Const)0f});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "for (int dim = 0; dim < A.width; ++dim)"});
            ptx.Add(new ld{ss = space.param, type = u32, d = a_width, a = a + 0});
            ptx.Add(new mov{type = u32, d = dim, a = (Const)0});
            ptx.Add(new setp{cmpop = cmp.le, type = u32, p = new Modded{Mod = Mod.Couple, Embedded = {p(2), p(8)}}, a = a_width, b = dim});
            ptx.Add(new bra{Guard = p(8).mod(Mod.Not), tgt = after_loop});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col])"});
            ptx.Add(new ld{ss = space.param, type = u32, d = a_raw, a = a + 8});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = r(18), a = a_width, b = row});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = a_offset_lo, a = r(18), b = (Const)4});
            ptx.Add(new add{type = u32, d = a_offset, a = a_offset_lo, b = a_raw});
            ptx.Add(new add{type = u32, d = r(21), a = r(18), b = a_width});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = r(25), a = r(21), b = (Const)4});
            ptx.Add(new add{type = u32, d = a_offset_hi, a = r(25), b = a_raw});
            ptx.Add(new ld{ss = space.param, type = u32, d = b_raw, a = b + 8});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = b_offset_lo, a = col, b = (Const)4});
            ptx.Add(new add{type = u32, d = b_offset, a = b_offset_lo, b = b_raw});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = b_offset_stride, a = b_width, b = (Const)4});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "// Cvalue += A.elements[row * A.width + dim] * B.elements[dim * B.width + col]"});
            ptx.Add(loop_body);
            ptx.Add(new ld{ss = space.global, type = f32, d = f(2), a = a_offset});
            ptx.Add(new ld{ss = space.global, type = f32, d = f(3), a = b_offset});
            ptx.Add(new mad{type = f32, d = cvalue, a = f(3), b = f(2), c = cvalue});
            ptx.Add(new add{type = u32, d = a_offset, a = a_offset, b = (Const)4});
            ptx.Add(new add{type = u32, d = b_offset, a = b_offset, b = b_offset_stride});
            ptx.Add(new setp{cmpop = cmp.ne, type = u32, p = p(3), a = a_offset, b = a_offset_hi});
            ptx.Add(new bra{Guard = p(3), tgt = loop_body});
            ptx.Add(new bra{uni = true, tgt = after_loop});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(new Comment{Text = "// C.elements[row * C.width + col] = Cvalue;"});
            ptx.Add(after_loop);
            ptx.Add(new ld{ss = space.param, type = u32, d = c_raw, a = c + 8});
            ptx.Add(new ld{ss = space.param, type = u32, d = c_width, a = c + 0});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = r(32), a = c_width, b = row});
            ptx.Add(new add{type = u32, d = r(33), a = col, b = r(32)});
            ptx.Add(new mul{mode = mulm.lo, type = u32, d = r(34), a = r(33), b = (Const)4});
            ptx.Add(new add{type = u32, d = r(35), a = c_raw, b = r(34)});
            ptx.Add(new st{ss = space.global, type = f32, a = r(35), b = cvalue});

            ptx.Add(new Comment{Text = Environment.NewLine});
            ptx.Add(exit);
            ptx.Add(new exit());

            module.Validate();
            var s_ptx = module.RenderAsPtx();
            VerifyResult(s_ptx);
        }
    }
}
