using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop("set.cmpop{.ftz}.dtype.stype         d, a, b;")]
    [Ptxop("set.cmpop.boolop{.ftz}.dtype.stype  d, a, b, {!}c;")]
    [Ptxop("setp.cmpop{.ftz}.stype              p[|q], a, b;")]
    [Ptxop("setp.cmpop.boolop{.ftz}.stype       p[|q], a, b, {!}c;")]
    [DebuggerNonUserCode]
    public class set : ptxop
    {
        [Mod] public bool p { get; set; }
        [Affix] public cmp cmpop { get; set; }
        [Affix] public op boolop { get; set; }
        [Affix] public bool ftz { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type stype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (boolop == null || boolop == and || boolop == or || boolop == xor).AssertTrue();
            (dtype == null || dtype == u32 || dtype == s32 || dtype == f32).AssertTrue();
            (p == true).AssertEquiv(dtype == null);
        }
    }
}