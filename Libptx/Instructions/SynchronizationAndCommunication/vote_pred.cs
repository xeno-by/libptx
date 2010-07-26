using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.SynchronizationAndCommunication
{
    [Ptxop("vote.mode.pred d, {!}a;", SoftwareIsa.PTX_12)]
    [DebuggerNonUserCode]
    public partial class vote_pred : ptxop
    {
        [Affix] public redm mode { get; set; }

        protected override void custom_validate_opcode(Module ctx)
        {
            (mode != 0).AssertTrue();
        }

        public vote_pred() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_alu(d, pred).AssertTrue();
            is_alu(a, pred, not).AssertTrue();
        }
    }
}