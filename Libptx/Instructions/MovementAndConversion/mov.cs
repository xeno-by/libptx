using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Expressions.Addresses;
using Libptx.Instructions.Annotations;
using Libptx.Expressions;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("mov.type d, a;")]
    [DebuggerNonUserCode]
    public partial class mov : ptxop
    {
        [Affix] public Type type { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_pred { get { return true; } }
        protected override bool allow_vec { get { return true; } }

        public mov() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_reg(d, type).AssertTrue();

            var move_from_alu_or_sreg = is_alu_or_sreg(a, type);
            var move_address = a is Address;
            var move_from_opaque_to_ptr = a.is_opaque() && (agree(type, u32) || agree(type, u64));
            (move_from_alu_or_sreg || move_address || move_from_opaque_to_ptr).AssertTrue();
        }
    }
}