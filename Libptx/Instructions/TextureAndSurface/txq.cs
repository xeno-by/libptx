using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;
using Libptx.Common.Enumerations;
using XenoGears.Assertions;
using XenoGears.Strings;
using Libptx.Expressions;
using XenoGears.Functional;
using Type=Libptx.Common.Types.Type;

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

        protected override void custom_validate_opcode()
        {
            (tquery != 0).AssertTrue();
            (type == b32).AssertTrue();
        }

        public txq() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        [Destination] public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands()
        {
            is_reg(d, type).AssertTrue();
            (is_texref(a) || is_samplerref(a) || agree(a, u32) || agree(a, u64)).AssertTrue();

            var is_tex_query = (tquery == tex_width || tquery == tex_height || tquery == tex_depth || tquery == tex_channel_datatype || tquery == tex_channel_order || tquery == tex_normalized_coords);
            var is_sampler_query = (tquery == tex_filter_mode || tquery == tex_addr_mode_0 || tquery == tex_addr_mode_1 || tquery == tex_addr_mode_2);
            is_tex_query.AssertImplies(is_texref(a) || agree(a, u32) || agree(a, u64));
            is_sampler_query.AssertImplies(is_samplerref(a) || (ctx.UnifiedTexturing && is_texref(a)) || agree(a, u32) || agree(a, u64));
        }

        protected override string custom_render_ptx(string core)
        {
            var iof = core.IndexOf(",");
            var before = core.Slice(0, iof);
            var after = core.Slice(iof + 2, -1);
            return String.Format("{0}, [{1}];", before, after);
        }
    }
}