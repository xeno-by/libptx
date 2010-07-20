using System;
using System.Linq;
using Libptx.Common.Enumerations;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libptx.Expressions;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_s16_v1 : var
    {
        public var_s16_v1_a1 this[int dim] { get { return Clone<var_s16_v1_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public var_s16 x { get { return Clone<var_s16>(v => v.Type = v.Type.x, v => v.Init = null); } }
        public var_s16 r { get { return Clone<var_s16>(v => v.Type = v.Type.r, v => v.Init = null); } }
        public new var_s16_v1 reg { get { return Clone(v => v.Space = space.reg); } }
        public new var_s16_v1 sreg { get { return Clone(v => v.Space = space.sreg); } }
        public new var_s16_v1 local { get { return Clone(v => v.Space = space.local); } }
        public new var_s16_v1 shared { get { return Clone(v => v.Space = space.shared); } }
        public new var_s16_v1 global { get { return Clone(v => v.Space = space.global); } }
        public new var_s16_v1 param { get { return Clone(v => v.Space = space.param); } }
        public new var_s16_v1 const0 { get { return Clone(v => v.Space = space.const0); } }
        public new var_s16_v1 const1 { get { return Clone(v => v.Space = space.const1); } }
        public new var_s16_v1 const2 { get { return Clone(v => v.Space = space.const2); } }
        public new var_s16_v1 const3 { get { return Clone(v => v.Space = space.const3); } }
        public new var_s16_v1 const4 { get { return Clone(v => v.Space = space.const4); } }
        public new var_s16_v1 const5 { get { return Clone(v => v.Space = space.const5); } }
        public new var_s16_v1 const6 { get { return Clone(v => v.Space = space.const6); } }
        public new var_s16_v1 const7 { get { return Clone(v => v.Space = space.const7); } }
        public new var_s16_v1 const8 { get { return Clone(v => v.Space = space.const8); } }
        public new var_s16_v1 const9 { get { return Clone(v => v.Space = space.const9); } }
        public new var_s16_v1 const10 { get { return Clone(v => v.Space = space.const10); } }

        public var_s16_v1 init(short1 value) { return Clone(v => v.Init = value); }

        public var_s16_v1() { Alignment = 2 /* sizeof(short1) */; }
        public var_s16_v1 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_s16_v1 align2{ get { return align(2); } }
        public var_s16_v1 align4{ get { return align(4); } }
        public var_s16_v1 align8{ get { return align(8); } }
        public var_s16_v1 align16{ get { return align(16); } }

        public var_s16_v1 export { get { return Clone(v => v.IsVisible = true); } }
        public var_s16_v1 import { get { return Clone(v => v.IsExtern = true); } }
        public var_s16_v1 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_s16_v1 Clone()
        {
            return Clone<var_s16_v1>();
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

        internal var_s16_v1 Clone(params Action<var_s16_v1>[] mods)
        {
            return Clone<var_s16_v1>(mods);
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
