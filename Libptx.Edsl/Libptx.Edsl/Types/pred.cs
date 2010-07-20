using System.Linq;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class pred : type
    {
        public static var_pred reg { get { return new var_pred().Clone(v => v.Space = space.reg); } }
        public static var_pred sreg { get { return new var_pred().Clone(v => v.Space = space.sreg); } }
        public static var_pred local { get { return new var_pred().Clone(v => v.Space = space.local); } }
        public static var_pred shared { get { return new var_pred().Clone(v => v.Space = space.shared); } }
        public static var_pred global { get { return new var_pred().Clone(v => v.Space = space.global); } }
        public static var_pred param { get { return new var_pred().Clone(v => v.Space = space.param); } }
        public static var_pred const0 { get { return new var_pred().Clone(v => v.Space = space.const0); } }
        public static var_pred const1 { get { return new var_pred().Clone(v => v.Space = space.const1); } }
        public static var_pred const2 { get { return new var_pred().Clone(v => v.Space = space.const2); } }
        public static var_pred const3 { get { return new var_pred().Clone(v => v.Space = space.const3); } }
        public static var_pred const4 { get { return new var_pred().Clone(v => v.Space = space.const4); } }
        public static var_pred const5 { get { return new var_pred().Clone(v => v.Space = space.const5); } }
        public static var_pred const6 { get { return new var_pred().Clone(v => v.Space = space.const6); } }
        public static var_pred const7 { get { return new var_pred().Clone(v => v.Space = space.const7); } }
        public static var_pred const8 { get { return new var_pred().Clone(v => v.Space = space.const8); } }
        public static var_pred const9 { get { return new var_pred().Clone(v => v.Space = space.const9); } }
        public static var_pred const10 { get { return new var_pred().Clone(v => v.Space = space.const10); } }

        public static var_pred init(bool value) { return new var_pred().Clone(v => v.Init = value); }

        public static var_pred align(int alignment){ return new var_pred().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_pred align4{ get { return align(4); } }
        public static var_pred align8{ get { return align(8); } }
        public static var_pred align16{ get { return align(16); } }
        public static var_pred align32{ get { return align(32); } }

        public static var_pred export { get { return new var_pred().Clone(v => v.IsVisible = true); } }
        public static var_pred import { get { return new var_pred().Clone(v => v.IsExtern = true); } }
        public static var_pred @extern { get { return new var_pred().Clone(v => v.IsExtern = true); } }
    }
}
