using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_f16 : has_type_f16
    {
        public var_f16_v1 v1 { get { return Clone<var_f16_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public var_f16_v2 v2 { get { return Clone<var_f16_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public var_f16_v4 v4 { get { return Clone<var_f16_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_f16_a1 this[int dim] { get { return Clone<var_f16_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new var_f16 reg { get { return Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public new var_f16 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public new var_f16 local { get { return Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public new var_f16 shared { get { return Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public new var_f16 global { get { return Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public new var_f16 param { get { return Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public new var_f16 @const { get { return Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public new var_f16 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public new var_f16 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public new var_f16 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public new var_f16 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public new var_f16 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public new var_f16 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public new var_f16 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public new var_f16 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public new var_f16 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public new var_f16 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public new var_f16 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public var_f16 init(half value) { return Clone(v => v.Init = value); }

        public var_f16() { Alignment = 2 /* sizeof(half) */; }
        public var_f16 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_f16 align2{ get { return align(2); } }
        public var_f16 align4{ get { return align(4); } }
        public var_f16 align8{ get { return align(8); } }
        public var_f16 align16{ get { return align(16); } }

        public var_f16 export { get { return Clone(v => v.IsVisible = true); } }
        public var_f16 import { get { return Clone(v => v.IsExtern = true); } }
        public var_f16 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_f16 Clone()
        {
            return Clone<var_f16>();
        }

        internal T Clone<T>()
            where T : var, new()
        {
            T clone = new T();
            clone.Name = this.Name;
            clone.Space = this.Space;
            clone.Type = this.Type;
            clone.Init = this.Init;
            clone.Alignment = this.Alignment;
            clone.Mod = this.Mod;
            clone.IsVisible = this.IsVisible;
            clone.IsExtern = this.IsExtern;
            return clone;
        }

        internal var_f16 Clone(params Action<var_f16>[] mods)
        {
            return Clone<var_f16>(mods);
        }

        internal T Clone<T>(params Action<T>[] mods)
            where T : var, new()
        {
            T clone = Clone<T>();
            foreach (Action<T> mod in mods) mod(clone);
            return clone;
        }
    }
}
