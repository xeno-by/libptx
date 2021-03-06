using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Slots;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using Type=Libptx.Common.Types.Type;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("cvta.space.size     p, a;")]
    [DebuggerNonUserCode]
    public partial class cvta : ptxop
    {
        [Affix] public space space { get; set; }
        [Affix] public Type size { get; set; }

        protected override void custom_validate_opcode()
        {
            (space == local || space == shared || space == global).AssertTrue();
            (size == u32 || size == u64).AssertTrue();
        }

        public cvta() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression p { get { return Operands[0]; } set { Operands[0] = value; } }
        public Address a { get { return (Address)Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(p, size).AssertTrue();

            is_ptr(a, space).AssertTrue();
            (a.Base == null).AssertTrue();
            if (a.Offset.Base is Reg) agree(a.Offset.Base, size).AssertTrue();
        }
    }
}