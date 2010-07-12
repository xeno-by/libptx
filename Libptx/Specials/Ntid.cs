using Libcuda.DataTypes;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Expressions;
using Libptx.Specials.Annotations;
using Type=Libptx.Common.Type;

namespace Libptx.Specials
{
    [Special10("%ntid", typeof(uint4))]
    public class Ntid : Special
    {
        protected override Type CustomType
        {
            get
            {
                var v4_u32 = new Type{Name = TypeName.U32, Mod = TypeMod.V4};
                var v4_u16 = new Type{Name = TypeName.U16, Mod = TypeMod.V4};
                return Ctx.Version >= SoftwareIsa.PTX_20 ? v4_u32 : v4_u16;
            }
        }
    }
}

