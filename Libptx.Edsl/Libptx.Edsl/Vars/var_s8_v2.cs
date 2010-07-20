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
    public class var_s8_v2 : var
    {
        public var_s8_v2_a1 this[int dim] { get { return Clone<var_s8_v2_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public var_s8 x { get { return Clone<var_s8>(v => v.Type = v.Type.x, v => v.Init = null); } }
        public var_s8 r { get { return Clone<var_s8>(v => v.Type = v.Type.r, v => v.Init = null); } }
        public var_s8 y { get { return Clone<var_s8>(v => v.Type = v.Type.y, v => v.Init = null); } }
        public var_s8 g { get { return Clone<var_s8>(v => v.Type = v.Type.g, v => v.Init = null); } }
        public new var_s8_v2 reg { get { return Clone(v => v.Space = space.reg); } }
        public new var_s8_v2 sreg { get { return Clone(v => v.Space = space.sreg); } }
        public new var_s8_v2 local { get { return Clone(v => v.Space = space.local); } }
        public new var_s8_v2 shared { get { return Clone(v => v.Space = space.shared); } }
        public new var_s8_v2 global { get { return Clone(v => v.Space = space.global); } }
        public new var_s8_v2 param { get { return Clone(v => v.Space = space.param); } }
        public new var_s8_v2 const0 { get { return Clone(v => v.Space = space.const0); } }
        public new var_s8_v2 const1 { get { return Clone(v => v.Space = space.const1); } }
        public new var_s8_v2 const2 { get { return Clone(v => v.Space = space.const2); } }
        public new var_s8_v2 const3 { get { return Clone(v => v.Space = space.const3); } }
        public new var_s8_v2 const4 { get { return Clone(v => v.Space = space.const4); } }
        public new var_s8_v2 const5 { get { return Clone(v => v.Space = space.const5); } }
        public new var_s8_v2 const6 { get { return Clone(v => v.Space = space.const6); } }
        public new var_s8_v2 const7 { get { return Clone(v => v.Space = space.const7); } }
        public new var_s8_v2 const8 { get { return Clone(v => v.Space = space.const8); } }
        public new var_s8_v2 const9 { get { return Clone(v => v.Space = space.const9); } }
        public new var_s8_v2 const10 { get { return Clone(v => v.Space = space.const10); } }

        public var_s8_v2 init(sbyte2 value) { return Clone(v => v.Init = value); }

        public var_s8_v2() { Alignment = 2 /* sizeof(sbyte2) */; }
        public var_s8_v2 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_s8_v2 align2{ get { return align(2); } }
        public var_s8_v2 align4{ get { return align(4); } }
        public var_s8_v2 align8{ get { return align(8); } }
        public var_s8_v2 align16{ get { return align(16); } }

        public var_s8_v2 export { get { return Clone(v => v.IsVisible = true); } }
        public var_s8_v2 import { get { return Clone(v => v.IsExtern = true); } }
        public var_s8_v2 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_s8_v2 Clone()
        {
            return Clone<var_s8_v2>();
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

        internal var_s8_v2 Clone(params Action<var_s8_v2>[] mods)
        {
            return Clone<var_s8_v2>(mods);
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
