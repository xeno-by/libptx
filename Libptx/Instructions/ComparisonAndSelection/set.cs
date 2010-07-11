using System.Diagnostics;
using Libcuda.Versions;
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
    internal class set : ptxop
    {
        [Endian] public bool p { get; set; }
        [Suffix] public cmpop cmpop { get; set; }
        [Suffix] public op boolop { get; set; }
        [Suffix] public bool ftz { get; set; }
        [Suffix] public type dtype { get; set; }
        [Suffix] public type stype { get; set; }

        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (boolop == null || boolop == and || boolop == or || boolop == xor).AssertTrue();
            (dtype == null || dtype == u32 || dtype == s32 || dtype == f32).AssertTrue();
            (p == true).AssertEquiv(dtype == null);
        }
    }
}