using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Libptx.Expressions;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("st{.ss}{.cop}.type          [a], b;")]
    [Ptxop("st.volatile{.ss}.type       [a], b;")]
    [DebuggerNonUserCode]
    public partial class st : ptxop
    {
        [Affix(SoftwareIsa.PTX_11)] public bool @volatile { get; set; }
        [Affix] public space ss { get; set; }
        [Affix] public cop cop { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var generic = ss == 0;
                var cache = cop != 0;
                return (generic || cache) ? SoftwareIsa.PTX_20 : SoftwareIsa.PTX_11;
            }
        }

        protected override HardwareIsa custom_hwisa
        {
            get
            {
                var generic = ss == 0;
                var cache = cop != 0;
                return (generic || cache) ? HardwareIsa.SM_20 : HardwareIsa.SM_10;
            }
        }

        protected override bool allow_int8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (@volatile == true).AssertEquiv(cop == 0);
            (cop == 0 || cop == wb || cop == cg || cop == cs || cop == wt).AssertTrue();
            type.is_v1().AssertFalse();
        }

        public Expression a { get; set; }
        public Expression b { get; set; }

        protected override bool allow_ptr { get { return true; } }
        protected override void custom_validate_operands(Module ctx)
        {
            is_ptr(a, ss).AssertTrue();
            (relax(b, type) && is_reg(b)).AssertTrue();
        }
    }
}