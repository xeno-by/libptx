using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("vote.mode.type d, {!}a;", SoftwareIsa.PTX_12)]
    [DebuggerNonUserCode]
    public partial class vote : ptxop
    {
        [Affix] public votem mode { get; set; }
        [Affix] public Type type { get; set; }

        protected override void custom_validate_opcode()
        {
            (mode == ballot || mode == all || mode == any || mode == uni).AssertTrue();
            (mode == ballot).AssertImplies(type == b32);
            (mode == all || mode == any || mode == uni).AssertImplies(type == pred);
        }

        public vote() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_alu(d, type).AssertTrue();
            is_alu(a, pred, not).AssertTrue();
        }
    }
}