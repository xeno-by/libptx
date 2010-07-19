using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class b8 : type
    {
        public static new var_b8_v1 v1 { get { return new var_b8().Clone<var_b8_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public static new var_b8_v2 v2 { get { return new var_b8().Clone<var_b8_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public static new var_b8_v4 v4 { get { return new var_b8().Clone<var_b8_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        // public static new var_b8_a1 this[int dim] { get { return new var_b8().Clone<var_b8_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public static var_b8 reg { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public static var_b8 sreg { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public static var_b8 local { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public static var_b8 shared { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public static var_b8 global { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public static var_b8 param { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public static var_b8 @const { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public static var_b8 const0 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public static var_b8 const1 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public static var_b8 const2 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public static var_b8 const3 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public static var_b8 const4 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public static var_b8 const5 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public static var_b8 const6 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public static var_b8 const7 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public static var_b8 const8 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public static var_b8 const9 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public static var_b8 const10 { get { return new var_b8().Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public static var_b8 init(Bit8 value) { return new var_b8().Clone(v => v.Init = value); }

        public static var_b8 align(int alignment){ return new var_b8().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_b8 align1{ get { return align(1); } }
        public static var_b8 align2{ get { return align(2); } }
        public static var_b8 align4{ get { return align(4); } }
        public static var_b8 align8{ get { return align(8); } }

        public static var_b8 export { get { return new var_b8().Clone(v => v.IsVisible = true); } }
        public static var_b8 import { get { return new var_b8().Clone(v => v.IsExtern = true); } }
        public static var_b8 @extern { get { return new var_b8().Clone(v => v.IsExtern = true); } }
    }
}
