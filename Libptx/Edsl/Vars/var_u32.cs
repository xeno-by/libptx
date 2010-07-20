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
    public class var_u32 : has_type_u32
    {
        public static var_u32 operator -(var_u32 var_u32) { return var_u32.Clone(v => v.Mod |= VarMod.Neg); }
        public var_u32 b0 { get { return Clone(v => v.Mod |= VarMod.B0); } }
        public var_u32 b1 { get { return Clone(v => v.Mod |= VarMod.B1); } }
        public var_u32 b2 { get { return Clone(v => v.Mod |= VarMod.B2); } }
        public var_u32 b3 { get { return Clone(v => v.Mod |= VarMod.B3); } }
        public var_u32 h0 { get { return Clone(v => v.Mod |= VarMod.H0); } }
        public var_u32 h1 { get { return Clone(v => v.Mod |= VarMod.H1); } }

        public var_u32_v1 v1 { get { return Clone<var_u32_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public var_u32_v2 v2 { get { return Clone<var_u32_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public var_u32_v4 v4 { get { return Clone<var_u32_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_u32_a1 this[int dim] { get { return Clone<var_u32_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public new var_u32 reg { get { return Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public new var_u32 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public new var_u32 local { get { return Clone(v => v.Space = Common.Enumerations.space.local); } }
        public new var_u32 shared { get { return Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public new var_u32 global { get { return Clone(v => v.Space = Common.Enumerations.space.global); } }
        public new var_u32 param { get { return Clone(v => v.Space = Common.Enumerations.space.param); } }
        public new var_u32 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public new var_u32 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public new var_u32 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public new var_u32 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public new var_u32 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public new var_u32 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public new var_u32 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public new var_u32 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public new var_u32 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public new var_u32 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public new var_u32 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.const10); } }

        public var_u32 init(uint value) { return Clone(v => v.Init = value); }

        public var_u32() { Alignment = 4 /* sizeof(uint) */; }
        public var_u32 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_u32 align4{ get { return align(4); } }
        public var_u32 align8{ get { return align(8); } }
        public var_u32 align16{ get { return align(16); } }
        public var_u32 align32{ get { return align(32); } }

        public var_u32 export { get { return Clone(v => v.IsVisible = true); } }
        public var_u32 import { get { return Clone(v => v.IsExtern = true); } }
        public var_u32 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_u32 Clone()
        {
            return Clone<var_u32>();
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

        internal var_u32 Clone(params Action<var_u32>[] mods)
        {
            return Clone<var_u32>(mods);
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
