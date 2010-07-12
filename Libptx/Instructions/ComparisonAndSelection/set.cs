using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.ComparisonAndSelection
{
    [Ptxop10("set.cmpop{.ftz}.dtype.stype         d, a, b;")]
    [Ptxop10("set.cmpop.boolop{.ftz}.dtype.stype  d, a, b, {!}c;")]
    [Ptxop10("setp.cmpop{.ftz}.stype              p[|q], a, b;")]
    [Ptxop10("setp.cmpop.boolop{.ftz}.stype       p[|q], a, b, {!}c;")]
    [DebuggerNonUserCode]
    internal class set : ptxop
    {
        [Mod] public bool p { get; set; }
        [Infix] public cmpop cmpop { get; set; }
        [Infix] public op boolop { get; set; }
        [Infix] public bool ftz { get; set; }
        [Infix] public type dtype { get; set; }
        [Infix] public type stype { get; set; }

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