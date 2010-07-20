using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_b8 : has_type_b8
    {
        public var_b8_v1 v1 { get { return Clone<var_b8_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public var_b8_v2 v2 { get { return Clone<var_b8_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public var_b8_v4 v4 { get { return Clone<var_b8_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_b8_a1 this[int dim] { get { return Clone<var_b8_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new var_b8 reg { get { return Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public new var_b8 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public new var_b8 local { get { return Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public new var_b8 shared { get { return Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public new var_b8 global { get { return Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public new var_b8 param { get { return Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public new var_b8 @const { get { return Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public new var_b8 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public new var_b8 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public new var_b8 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public new var_b8 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public new var_b8 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public new var_b8 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public new var_b8 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public new var_b8 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public new var_b8 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public new var_b8 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public new var_b8 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public var_b8 init(Bit8 value) { return Clone(v => v.Init = value); }

        public var_b8() { Alignment = 1 /* sizeof(Bit8) */; }
        public var_b8 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_b8 align1{ get { return align(1); } }
        public var_b8 align2{ get { return align(2); } }
        public var_b8 align4{ get { return align(4); } }
        public var_b8 align8{ get { return align(8); } }

        public var_b8 export { get { return Clone(v => v.IsVisible = true); } }
        public var_b8 import { get { return Clone(v => v.IsExtern = true); } }
        public var_b8 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_b8 Clone()
        {
            return Clone<var_b8>();
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

        internal var_b8 Clone(params Action<var_b8>[] mods)
        {
            return Clone<var_b8>(mods);
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
