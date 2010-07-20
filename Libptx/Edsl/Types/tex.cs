using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class tex : type
    {
        public static var_tex reg { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public static var_tex sreg { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public static var_tex local { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.local); } }
        public static var_tex shared { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public static var_tex global { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.global); } }
        public static var_tex param { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.param); } }
        public static var_tex const0 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public static var_tex const1 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public static var_tex const2 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public static var_tex const3 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public static var_tex const4 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public static var_tex const5 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public static var_tex const6 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public static var_tex const7 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public static var_tex const8 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public static var_tex const9 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public static var_tex const10 { get { return new var_tex().Clone(v => v.Space = Common.Enumerations.space.const10); } }

        public static var_tex init(Tex value) { return new var_tex().Clone(v => v.Init = value); }

        public static var_tex align(int alignment){ return new var_tex().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_tex align1{ get { return align(1); } }
        public static var_tex align2{ get { return align(2); } }
        public static var_tex align4{ get { return align(4); } }
        public static var_tex align8{ get { return align(8); } }

        public static var_tex export { get { return new var_tex().Clone(v => v.IsVisible = true); } }
        public static var_tex import { get { return new var_tex().Clone(v => v.IsExtern = true); } }
        public static var_tex @extern { get { return new var_tex().Clone(v => v.IsExtern = true); } }
    }
}
