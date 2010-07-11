using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Instructions.Annotations;
using Libptx.Instructions.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("suq.query.b32 d, [a];", SoftwareIsa.PTX_15)]
    [DebuggerNonUserCode]
    internal class suq : ptxop
    {
        [Suffix] public squery query { get; set; }
        [Suffix] public type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cdt_or_co_query = query == surf_channel_datatype || query == surf_channel_order;
                return cdt_or_co_query ? SoftwareIsa.PTX_21 : SoftwareIsa.PTX_15;
            }
        }

        protected override void custom_validate(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (query != null).AssertTrue();
            (type == b32).AssertTrue();
        }
    }
}