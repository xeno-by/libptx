using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop("suq.query.b32 d, [a];", SoftwareIsa.PTX_15)]
    [DebuggerNonUserCode]
    public partial class suq : ptxop
    {
        [Affix] public squery query { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cdt_or_co_query = query == surf_channel_datatype || query == surf_channel_order;
                return cdt_or_co_query ? SoftwareIsa.PTX_21 : SoftwareIsa.PTX_15;
            }
        }

        protected override void custom_validate_opcode(SoftwareIsa target_swisa, HardwareIsa target_hwisa)
        {
            (query != 0).AssertTrue();
            (type == b32).AssertTrue();
        }
    }
}