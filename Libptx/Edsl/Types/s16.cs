using System.Linq;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class s16 : type
    {
        public static new var_s16_v1 v1 { get { return new var_s16().Clone<var_s16_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public static new var_s16_v2 v2 { get { return new var_s16().Clone<var_s16_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public static new var_s16_v4 v4 { get { return new var_s16().Clone<var_s16_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_s16_a1 this[int dim] { get { return new var_s16().Clone<var_s16_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public static var_s16 reg { get { return new var_s16().Clone(v => v.Space = space.reg); } }
        public static var_s16 sreg { get { return new var_s16().Clone(v => v.Space = space.sreg); } }
        public static var_s16 local { get { return new var_s16().Clone(v => v.Space = space.local); } }
        public static var_s16 shared { get { return new var_s16().Clone(v => v.Space = space.shared); } }
        public static var_s16 global { get { return new var_s16().Clone(v => v.Space = space.global); } }
        public static var_s16 param { get { return new var_s16().Clone(v => v.Space = space.param); } }
        public static var_s16 const0 { get { return new var_s16().Clone(v => v.Space = space.const0); } }
        public static var_s16 const1 { get { return new var_s16().Clone(v => v.Space = space.const1); } }
        public static var_s16 const2 { get { return new var_s16().Clone(v => v.Space = space.const2); } }
        public static var_s16 const3 { get { return new var_s16().Clone(v => v.Space = space.const3); } }
        public static var_s16 const4 { get { return new var_s16().Clone(v => v.Space = space.const4); } }
        public static var_s16 const5 { get { return new var_s16().Clone(v => v.Space = space.const5); } }
        public static var_s16 const6 { get { return new var_s16().Clone(v => v.Space = space.const6); } }
        public static var_s16 const7 { get { return new var_s16().Clone(v => v.Space = space.const7); } }
        public static var_s16 const8 { get { return new var_s16().Clone(v => v.Space = space.const8); } }
        public static var_s16 const9 { get { return new var_s16().Clone(v => v.Space = space.const9); } }
        public static var_s16 const10 { get { return new var_s16().Clone(v => v.Space = space.const10); } }

        public static var_s16 init(short value) { return new var_s16().Clone(v => v.Init = value); }

        public static var_s16 align(int alignment){ return new var_s16().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_s16 align2{ get { return align(2); } }
        public static var_s16 align4{ get { return align(4); } }
        public static var_s16 align8{ get { return align(8); } }
        public static var_s16 align16{ get { return align(16); } }

        public static var_s16 export { get { return new var_s16().Clone(v => v.IsVisible = true); } }
        public static var_s16 import { get { return new var_s16().Clone(v => v.IsExtern = true); } }
        public static var_s16 @extern { get { return new var_s16().Clone(v => v.IsExtern = true); } }
    }
}
