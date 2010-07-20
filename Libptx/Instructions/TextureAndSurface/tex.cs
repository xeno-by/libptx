using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("tex.geom.v4.dtype.btype d, [a, c];")]
    [Ptxop("tex.geom.v4.dtype.btype d, [a, b, c];")]
    [DebuggerNonUserCode]
    public class tex : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public vec vec { get; set; }
        [Affix] public type dtype { get; set; }
        [Affix] public type btype { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (geom != null).AssertTrue();
            (vec == v4).AssertTrue();
            (dtype == u32 || dtype == s32 || dtype == f32).AssertTrue();
            (btype == s32 || btype == f32).AssertTrue();

            // todo. implement this:
            // Extension using opaque texref and samplerref types and independent mode texturing
            // introduced in PTX ISA version 1.5 (an excerpt from PTX ISA 2.1 manual)
        }
    }
}