using System.Diagnostics;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop20("sured.b.op.geom.ctype.clampm [a,b],c;")]
    [Ptxop20("sured.p.op.geom.ctype.clampm [a,b],c;")]
    [DebuggerNonUserCode]
    internal class sured : ptxop
    {
        [Suffix] public bool b { get; set; }
        [Suffix] public bool p { get; set; }
        [Suffix] public op op { get; set; }
        [Suffix] public geom geom { get; set; }
        [Suffix] public type ctype { get; set; }
        [Suffix] public clampm clampm { get; set; }

        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (b || p).AssertTrue();
            (geom != null).AssertTrue();
            (b == true).AssertImplies(ctype == b32);
            (p == true).AssertImplies(ctype.is32());
        }
    }
}