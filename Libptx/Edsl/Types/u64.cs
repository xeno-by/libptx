using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class u64 : type
    {
        public static new var_u64_v1 v1 { get { return new var_u64().Clone<var_u64_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public static new var_u64_v2 v2 { get { return new var_u64().Clone<var_u64_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public static new var_u64_v4 v4 { get { return new var_u64().Clone<var_u64_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        // public static new var_u64_a1 this[int dim] { get { return new var_u64().Clone<var_u64_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public static var_u64 reg { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public static var_u64 sreg { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public static var_u64 local { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public static var_u64 shared { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public static var_u64 global { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public static var_u64 param { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public static var_u64 @const { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public static var_u64 const0 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public static var_u64 const1 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public static var_u64 const2 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public static var_u64 const3 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public static var_u64 const4 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public static var_u64 const5 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public static var_u64 const6 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public static var_u64 const7 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public static var_u64 const8 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public static var_u64 const9 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public static var_u64 const10 { get { return new var_u64().Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public static var_u64 init(ulong value) { return new var_u64().Clone(v => v.Init = value); }

        public static var_u64 align(int alignment){ return new var_u64().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_u64 align8{ get { return align(8); } }
        public static var_u64 align16{ get { return align(16); } }
        public static var_u64 align32{ get { return align(32); } }
        public static var_u64 align64{ get { return align(64); } }

        public static var_u64 export { get { return new var_u64().Clone(v => v.IsVisible = true); } }
        public static var_u64 import { get { return new var_u64().Clone(v => v.IsExtern = true); } }
        public static var_u64 @extern { get { return new var_u64().Clone(v => v.IsExtern = true); } }
    }
}
