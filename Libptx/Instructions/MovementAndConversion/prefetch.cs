using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Expressions.Addresses;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop20("prefetch{.space}.level  [a];")]
    [Ptxop20("prefetchu.L1            [a];")]
    [DebuggerNonUserCode]
    public partial class prefetch : ptxop
    {
        [Mod] public bool u { get; set; }
        [Affix] public space space { get; set; }
        [Affix] public cachelevel level { get; set; }

        protected override void custom_validate_opcode()
        {
            (u == true).AssertEquiv(space == 0 && level == L1);
            (space == 0 || space == local || space == global).AssertTrue();
            (level == L1 || level == L2).AssertTrue();
        }

        public prefetch() { 1.UpTo(1).ForEach(_ => Operands.Add(null)); }
        public Address a { get { return (Address)Operands[0]; } set { Operands[0] = value; } }

        protected override void custom_validate_operands()
        {
            is_ptr(a, space).AssertTrue();
        }
    }
}