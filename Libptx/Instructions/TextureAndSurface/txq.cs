using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Infrastructure;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop10("txq.tquery.b32 d, [a];", SoftwareIsa.PTX_15)]
    [Ptxop10("txq.squery.b32 d, [a];", SoftwareIsa.PTX_15)]
    [DebuggerNonUserCode]
    internal class txq : ptxop
    {
        [Infix] public tquery tquery { get; set; }
        [Infix] public tquerys squery { get; set; }
        [Infix] public type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cdt_or_co_query = tquery == tex_channel_datatype || tquery == tex_channel_order;
                return cdt_or_co_query ? SoftwareIsa.PTX_21 : SoftwareIsa.PTX_15;
            }
        }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (tquery != null ^ squery != null).AssertTrue();
            (type == b32).AssertTrue();
        }
    }
}