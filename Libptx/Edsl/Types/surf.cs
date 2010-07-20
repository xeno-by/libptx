using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class surf : type
    {
        public static var_surf reg { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public static var_surf sreg { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public static var_surf local { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public static var_surf shared { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public static var_surf global { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public static var_surf param { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public static var_surf @const { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public static var_surf const0 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public static var_surf const1 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public static var_surf const2 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public static var_surf const3 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public static var_surf const4 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public static var_surf const5 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public static var_surf const6 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public static var_surf const7 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public static var_surf const8 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public static var_surf const9 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public static var_surf const10 { get { return new var_surf().Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public static var_surf init(Surf value) { return new var_surf().Clone(v => v.Init = value); }

        public static var_surf align(int alignment){ return new var_surf().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_surf align1{ get { return align(1); } }
        public static var_surf align2{ get { return align(2); } }
        public static var_surf align4{ get { return align(4); } }
        public static var_surf align8{ get { return align(8); } }

        public static var_surf export { get { return new var_surf().Clone(v => v.IsVisible = true); } }
        public static var_surf import { get { return new var_surf().Clone(v => v.IsExtern = true); } }
        public static var_surf @extern { get { return new var_surf().Clone(v => v.IsExtern = true); } }
    }
}
