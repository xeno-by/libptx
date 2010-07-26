using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using Libptx.Expressions;
using XenoGears.Functional;

namespace Libptx.Instructions.TextureAndSurface
{
    [Ptxop15("txq.tquery.b32 d, [a];")]
    [DebuggerNonUserCode]
    public partial class txq : ptxop
    {
        [Affix] public tquery tquery { get; set; }
        [Affix] public Type type { get; set; }

        protected override SoftwareIsa custom_swisa
        {
            get
            {
                var cdt_or_co_query = tquery == tex_channel_datatype || tquery == tex_channel_order;
                return cdt_or_co_query ? SoftwareIsa.PTX_21 : SoftwareIsa.PTX_15;
            }
        }

        protected override void custom_validate_opcode(Module ctx)
        {
            (tquery != 0).AssertTrue();
            (type == b32).AssertTrue();
        }

        public txq() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            agree(d, type).AssertTrue();
            (agree(a, texref) || agree(a, samplerref)).AssertTrue();

            var is_tex_query = (tquery == tex_width || tquery == tex_height || tquery == tex_depth || tquery == tex_channel_datatype || tquery == tex_channel_order || tquery == tex_normalized_coords);
            var is_sampler_query = (tquery == tex_filter_mode || tquery == tex_addr_mode_0 || tquery == tex_addr_mode_1 || tquery == tex_addr_mode_2);
            is_tex_query.AssertImplies(agree(a, texref));
            is_sampler_query.AssertImplies(agree(a, samplerref) || (ctx.UnifiedTexturing && agree(a, texref)));
        }
    }
}