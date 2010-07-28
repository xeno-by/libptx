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
    [Ptxop15("suq.query.b32 d, [a];")]
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

        protected override void custom_validate_opcode(Module ctx)
        {
            (query != 0).AssertTrue();
            (type == b32).AssertTrue();
        }

        public suq() { 1.UpTo(2).ForEach(_ => Operands.Add(null)); }
        public Expression d { get { return Operands[0]; } set { Operands[0] = value; } }
        public Expression a { get { return Operands[1]; } set { Operands[1] = value; } }

        protected override void custom_validate_operands(Module ctx)
        {
            is_reg(d, type).AssertTrue();
            (is_surfref(a) || agree(a, u32) || agree(a, u64)).AssertTrue();
        }
    }
}