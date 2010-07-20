using System.Linq;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class surf : type
    {
        public static var_surf reg { get { return new var_surf().Clone(v => v.Space = space.reg); } }
        public static var_surf sreg { get { return new var_surf().Clone(v => v.Space = space.sreg); } }
        public static var_surf local { get { return new var_surf().Clone(v => v.Space = space.local); } }
        public static var_surf shared { get { return new var_surf().Clone(v => v.Space = space.shared); } }
        public static var_surf global { get { return new var_surf().Clone(v => v.Space = space.global); } }
        public static var_surf param { get { return new var_surf().Clone(v => v.Space = space.param); } }
        public static var_surf const0 { get { return new var_surf().Clone(v => v.Space = space.const0); } }
        public static var_surf const1 { get { return new var_surf().Clone(v => v.Space = space.const1); } }
        public static var_surf const2 { get { return new var_surf().Clone(v => v.Space = space.const2); } }
        public static var_surf const3 { get { return new var_surf().Clone(v => v.Space = space.const3); } }
        public static var_surf const4 { get { return new var_surf().Clone(v => v.Space = space.const4); } }
        public static var_surf const5 { get { return new var_surf().Clone(v => v.Space = space.const5); } }
        public static var_surf const6 { get { return new var_surf().Clone(v => v.Space = space.const6); } }
        public static var_surf const7 { get { return new var_surf().Clone(v => v.Space = space.const7); } }
        public static var_surf const8 { get { return new var_surf().Clone(v => v.Space = space.const8); } }
        public static var_surf const9 { get { return new var_surf().Clone(v => v.Space = space.const9); } }
        public static var_surf const10 { get { return new var_surf().Clone(v => v.Space = space.const10); } }

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
