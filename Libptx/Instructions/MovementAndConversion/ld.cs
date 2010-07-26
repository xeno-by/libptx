using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Spaces;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libcuda.Versions;
using XenoGears.Assertions;
using Type=Libptx.Common.Types.Type;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.MovementAndConversion
{
    [Ptxop("ld{.ss}{.cop}.type          d, [a];")]
    [Ptxop("ld.volatile{.ss}.type       d, [a];")]
    [Ptxop("ldu{.ss}.type               d, [a];")]
    [DebuggerNonUserCode]
    public partial class ld : ptxop
    {
        [Mod] public bool u { get; set; }
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
        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override bool allow_vec { get { return true; } }
        protected override void custom_validate_opcode(Module ctx)
        {
            (u == true).AssertImplies(ss == 0 || ss == global);
            (u == true).AssertImplies(@volatile == false);
            (u == true).AssertImplies(cop == 0);
            (@volatile == true).AssertEquiv(cop == 0);
            (cop == 0 || cop == ca || cop == cg || cop == cs || cop == lu || cop == cv).AssertTrue();
            type.is_v1().AssertFalse();
        }

        public ld() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override bool allow_ptr { get { return true; } }
        protected override void custom_validate_operands(Module ctx)
        {
            (relax(d, type) && is_reg(d)).AssertTrue();
            is_ptr(a, ss).AssertTrue();
        }
    }
}