using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_s64_v1 : var
    {
        public var_s64_v1_a1 this[int dim] { get { return Clone<var_s64_v1_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public var_s64 x { get { return Clone<var_s64>(v => v.Type = v.Type.x, v => v.Init = null); } }
        public var_s64 r { get { return Clone<var_s64>(v => v.Type = v.Type.r, v => v.Init = null); } }

        public new var_s64_v1 reg { get { return Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public new var_s64_v1 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public new var_s64_v1 local { get { return Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public new var_s64_v1 shared { get { return Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public new var_s64_v1 global { get { return Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public new var_s64_v1 param { get { return Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public new var_s64_v1 @const { get { return Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public new var_s64_v1 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public new var_s64_v1 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public new var_s64_v1 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public new var_s64_v1 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public new var_s64_v1 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public new var_s64_v1 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public new var_s64_v1 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public new var_s64_v1 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public new var_s64_v1 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public new var_s64_v1 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public new var_s64_v1 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.Const10); } }

        public var_s64_v1 init(long1 value) { return Clone(v => v.Init = value); }

        public var_s64_v1() { Alignment = 8 /* sizeof(long1) */; }
        public var_s64_v1 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_s64_v1 align8{ get { return align(8); } }
        public var_s64_v1 align16{ get { return align(16); } }
        public var_s64_v1 align32{ get { return align(32); } }
        public var_s64_v1 align64{ get { return align(64); } }

        public var_s64_v1 export { get { return Clone(v => v.IsVisible = true); } }
        public var_s64_v1 import { get { return Clone(v => v.IsExtern = true); } }
        public var_s64_v1 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_s64_v1 Clone()
        {
            return Clone<var_s64_v1>();
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

        internal var_s64_v1 Clone(params Action<var_s64_v1>[] mods)
        {
            return Clone<var_s64_v1>(mods);
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
