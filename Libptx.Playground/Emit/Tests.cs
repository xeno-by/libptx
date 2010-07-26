using System;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Expressions.Slots;
using Libptx.Expressions.Slots.Specials;
using Libptx.Instructions.Arithmetic;
using Libptx.Instructions.MovementAndConversion;
using Libptx.Statements;
using NUnit.Framework;
using XenoGears.Playground.Framework;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Playground.Emit
{
    [TestFixture]
    public class Tests : BaseTests
    {
        [Test]
        public void MatMul()
        {
            Func<String, Var> reg_u32 = name => new Var{Name = name, Space = space.reg, Type = new Type{Name = TypeName.U32}};
            Func<String, Var> reg_f32 = name => new Var{Name = name, Space = space.reg, Type = new Type{Name = TypeName.F32}};
            Func<int, Var> rh = i => new Var{Name = String.Format("%rh{0}", i), Space = space.reg, Type = new Type{Name = TypeName.S16}};
            Func<int, Var> rhu = i => new Var{Name = String.Format("%rhu{0}", i), Space = space.reg, Type = new Type{Name = TypeName.U16}};
            Func<int, Var> r = i => new Var{Name = String.Format("%r{0}", i), Space = space.reg, Type = new Type{Name = TypeName.S32}};
            Func<int, Var> ru = i => new Var{Name = String.Format("%ru{0}", i), Space = space.reg, Type = new Type{Name = TypeName.U32}};
            Func<int, Var> rl = i => new Var{Name = String.Format("%rl{0}", i), Space = space.reg, Type = new Type{Name = TypeName.S64}};
            Func<int, Var> rlu = i => new Var{Name = String.Format("%rlu{0}", i), Space = space.reg, Type = new Type{Name = TypeName.U64}};
            Func<int, Var> f = i => new Var{Name = String.Format("%f{0}", i), Space = space.reg, Type = new Type{Name = TypeName.F32}};
            Func<int, Var> d = i => new Var{Name = String.Format("%d{0}", i), Space = space.reg, Type = new Type{Name = TypeName.F64}};
            Type u16 = new Type { Name = TypeName.U16 }, s16 = new Type { Name = TypeName.S16 };
            Type u32 = new Type { Name = TypeName.U32 }, s32 = new Type { Name = TypeName.S32 };
            Type f32 = new Type { Name = TypeName.F32 }, f64 = new Type { Name = TypeName.F64 };
            Type pred = new Type { Name = TypeName.Pred };

            var module = new Module(SoftwareIsa.PTX_14, HardwareIsa.SM_13);
            Func<String, Var> param_align4_b8_12 = name => new Var{Name = name, Space = space.param, Alignment = 4, Type = new Type{Name = TypeName.B8, Dims = new []{12}}};
            Var a = param_align4_b8_12("A"), b = param_align4_b8_12("B"), c = param_align4_b8_12("C");
            var kernel = module.AddEntry("MatMulKernel", a, b, c);
            var ptx = kernel.Body.Stmts;

            // todo. auto-add registers to a block if they ain't belong to any of parent blocks
            // todo. B {B1, B2}: if var is added to B1 first and then to B2, then lift it to B
            // todo. provide a way to opt-out from this behavior
            // todo. verify that names within a block do not collide
            Func<String, Label> label = name => new Label{Name = name};
            Label loop_body = label("$LoopBody"), after_loop = label("$AfterLoop"), exit = label("$Exit");
            Var a_width = reg_u32("a_width"), a_height = reg_u32("a_height"), a_raw = reg_u32("a_raw");
            Var b_width = reg_u32("b_width"), b_height = reg_u32("b_height"), b_raw = reg_u32("b_raw");
            Var c_width = reg_u32("c_width"), c_height = reg_u32("c_height"), c_raw = reg_u32("c_raw");
            Var row = reg_u32("row"), col = reg_u32("col"), cvalue = reg_f32("cvalue"), dim = reg_u32("dim");
            Var a_offset = reg_u32("a_offset"), a_offset_lo = reg_u32("a_offset_lo"), a_offset_stride = reg_u32("a_offset_stride"), a_offset_hi = reg_u32("a_offset_hi");
            Var b_offset = reg_u32("b_offset"), b_offset_lo = reg_u32("b_offset_lo"), b_offset_stride = reg_u32("b_offset_stride"), b_offset_hi = reg_u32("b_offset_hi");

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

            ptx.Add(new Comment{Text = "if (A.height <= row || B.width <= col) return;"});
            ptx.Add(new ld{ss = space.param, type = u32, d = b_width, a = b + 0});
        }
    }
}
