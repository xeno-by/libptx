using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Types
{
    public class sampler : type
    {
        public static var_sampler reg { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public static var_sampler sreg { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public static var_sampler local { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public static var_sampler shared { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public static var_sampler global { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public static var_sampler param { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public static var_sampler @const { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public static var_sampler const0 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public static var_sampler const1 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public static var_sampler const2 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public static var_sampler const3 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public static var_sampler const4 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public static var_sampler const5 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public static var_sampler const6 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public static var_sampler const7 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public static var_sampler const8 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public static var_sampler const9 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public static var_sampler const10 { get { return new var_sampler().Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public static var_sampler init(Sampler value) { return new var_sampler().Clone(v => v.Init = value); }

        public static var_sampler align(int alignment){ return new var_sampler().Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public static var_sampler align1{ get { return align(1); } }
        public static var_sampler align2{ get { return align(2); } }
        public static var_sampler align4{ get { return align(4); } }
        public static var_sampler align8{ get { return align(8); } }

        public static var_sampler export { get { return new var_sampler().Clone(v => v.IsVisible = true); } }
        public static var_sampler import { get { return new var_sampler().Clone(v => v.IsExtern = true); } }
        public static var_sampler @extern { get { return new var_sampler().Clone(v => v.IsExtern = true); } }
    }
}
