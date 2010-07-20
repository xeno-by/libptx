using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("tex.geom.dtype.btype d, [a, c];")]
    [Ptxop("tex.geom.dtype.btype d, [a, b, c];")]
    [DebuggerNonUserCode]
    public class tex : ptxop
    {
        [Affix] public geom geom { get; set; }
        [Affix] public Type dtype { get; set; }
        [Affix] public Type btype { get; set; }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (geom != 0).AssertTrue();
            (dtype.is32() && dtype.isv4()).AssertTrue();
            (btype == s32 || btype == f32).AssertTrue();

            // todo. implement this:
            // Extension using opaque texref and samplerref types and independent mode texturing
            // introduced in PTX ISA version 1.5 (an excerpt from PTX ISA 2.1 manual)
        }
    }
}