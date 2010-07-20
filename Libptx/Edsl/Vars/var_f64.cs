using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libptx.Expressions;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_f64 : has_type_f64
    {
        public var_f64_v1 v1 { get { return Clone<var_f64_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public var_f64_v2 v2 { get { return Clone<var_f64_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public var_f64_v4 v4 { get { return Clone<var_f64_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_f64_a1 this[int dim] { get { return Clone<var_f64_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public new var_f64 reg { get { return Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public new var_f64 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public new var_f64 local { get { return Clone(v => v.Space = Common.Enumerations.space.local); } }
        public new var_f64 shared { get { return Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public new var_f64 global { get { return Clone(v => v.Space = Common.Enumerations.space.global); } }
        public new var_f64 param { get { return Clone(v => v.Space = Common.Enumerations.space.param); } }
        public new var_f64 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public new var_f64 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public new var_f64 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public new var_f64 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public new var_f64 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public new var_f64 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public new var_f64 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public new var_f64 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public new var_f64 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public new var_f64 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public new var_f64 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.const10); } }

        public var_f64 init(double value) { return Clone(v => v.Init = value); }

        public var_f64() { Alignment = 8 /* sizeof(double) */; }
        public var_f64 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_f64 align8{ get { return align(8); } }
        public var_f64 align16{ get { return align(16); } }
        public var_f64 align32{ get { return align(32); } }
        public var_f64 align64{ get { return align(64); } }

        public var_f64 export { get { return Clone(v => v.IsVisible = true); } }
        public var_f64 import { get { return Clone(v => v.IsExtern = true); } }
        public var_f64 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_f64 Clone()
        {
            return Clone<var_f64>();
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

        internal var_f64 Clone(params Action<var_f64>[] mods)
        {
            return Clone<var_f64>(mods);
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
