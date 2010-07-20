using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class b64 : type
    {
        public static new var_b64_v1 v1 { get { return new var_b64().Clone<var_b64_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public static new var_b64_v2 v2 { get { return new var_b64().Clone<var_b64_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public static new var_b64_v4 v4 { get { return new var_b64().Clone<var_b64_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        // public static new var_b64_a1 this[int dim] { get { return new var_b64().Clone<var_b64_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public static var_b64 reg { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public static var_b64 sreg { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public static var_b64 local { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.local); } }
        public static var_b64 shared { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public static var_b64 global { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.global); } }
        public static var_b64 param { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.param); } }
        public static var_b64 const0 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public static var_b64 const1 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public static var_b64 const2 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public static var_b64 const3 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public static var_b64 const4 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public static var_b64 const5 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public static var_b64 const6 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public static var_b64 const7 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public static var_b64 const8 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public static var_b64 const9 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public static var_b64 const10 { get { return new var_b64().Clone(v => v.Space = Common.Enumerations.space.const10); } }

        public static var_b64 init(Bit64 value) { return new var_b64().Clone(v => v.Init = value); }

        public static var_b64 align(int alignment){ return new var_b64().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_b64 align8{ get { return align(8); } }
        public static var_b64 align16{ get { return align(16); } }
        public static var_b64 align32{ get { return align(32); } }
        public static var_b64 align64{ get { return align(64); } }

        public static var_b64 export { get { return new var_b64().Clone(v => v.IsVisible = true); } }
        public static var_b64 import { get { return new var_b64().Clone(v => v.IsExtern = true); } }
        public static var_b64 @extern { get { return new var_b64().Clone(v => v.IsExtern = true); } }
    }
}
