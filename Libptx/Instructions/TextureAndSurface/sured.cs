using System.Diagnostics;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop20("sured.b.op.geom.ctype.clampm [a,b],c;")]
    [Ptxop20("sured.p.op.geom.ctype.clampm [a,b],c;")]
    [DebuggerNonUserCode]
    public class sured : ptxop
    {
        [Affix] public bool b { get; set; }
        [Affix] public bool p { get; set; }
        [Affix] public op op { get; set; }
        [Affix] public geom geom { get; set; }
        [Affix] public Type ctype { get; set; }
        [Affix] public clampm clampm { get; set; }

        protected override bool allow_bit8 { get { return true; } }
        protected override bool allow_bit16 { get { return true; } }
        protected override bool allow_bit32 { get { return true; } }
        protected override bool allow_bit64 { get { return true; } }
        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (b || p).AssertTrue();
            (geom != 0).AssertTrue();
            (b == true).AssertImplies(ctype == b32);
            (p == true).AssertImplies(ctype.is32());
            (op == add || op == min || op == max || op == and || op == or || op == xor).AssertTrue();
        }
    }
}