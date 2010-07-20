using System.Linq;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class u8 : type
    {
        public static new var_u8_v1 v1 { get { return new var_u8().Clone<var_u8_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public static new var_u8_v2 v2 { get { return new var_u8().Clone<var_u8_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public static new var_u8_v4 v4 { get { return new var_u8().Clone<var_u8_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_u8_a1 this[int dim] { get { return new var_u8().Clone<var_u8_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public static var_u8 reg { get { return new var_u8().Clone(v => v.Space = space.reg); } }
        public static var_u8 sreg { get { return new var_u8().Clone(v => v.Space = space.sreg); } }
        public static var_u8 local { get { return new var_u8().Clone(v => v.Space = space.local); } }
        public static var_u8 shared { get { return new var_u8().Clone(v => v.Space = space.shared); } }
        public static var_u8 global { get { return new var_u8().Clone(v => v.Space = space.global); } }
        public static var_u8 param { get { return new var_u8().Clone(v => v.Space = space.param); } }
        public static var_u8 const0 { get { return new var_u8().Clone(v => v.Space = space.const0); } }
        public static var_u8 const1 { get { return new var_u8().Clone(v => v.Space = space.const1); } }
        public static var_u8 const2 { get { return new var_u8().Clone(v => v.Space = space.const2); } }
        public static var_u8 const3 { get { return new var_u8().Clone(v => v.Space = space.const3); } }
        public static var_u8 const4 { get { return new var_u8().Clone(v => v.Space = space.const4); } }
        public static var_u8 const5 { get { return new var_u8().Clone(v => v.Space = space.const5); } }
        public static var_u8 const6 { get { return new var_u8().Clone(v => v.Space = space.const6); } }
        public static var_u8 const7 { get { return new var_u8().Clone(v => v.Space = space.const7); } }
        public static var_u8 const8 { get { return new var_u8().Clone(v => v.Space = space.const8); } }
        public static var_u8 const9 { get { return new var_u8().Clone(v => v.Space = space.const9); } }
        public static var_u8 const10 { get { return new var_u8().Clone(v => v.Space = space.const10); } }

        public static var_u8 init(byte value) { return new var_u8().Clone(v => v.Init = value); }

        public static var_u8 align(int alignment){ return new var_u8().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_u8 align1{ get { return align(1); } }
        public static var_u8 align2{ get { return align(2); } }
        public static var_u8 align4{ get { return align(4); } }
        public static var_u8 align8{ get { return align(8); } }

        public static var_u8 export { get { return new var_u8().Clone(v => v.IsVisible = true); } }
        public static var_u8 import { get { return new var_u8().Clone(v => v.IsExtern = true); } }
        public static var_u8 @extern { get { return new var_u8().Clone(v => v.IsExtern = true); } }
    }
}
