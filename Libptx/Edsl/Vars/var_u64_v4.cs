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
    public class var_u64_v4 : var
    {
        public var_u64_v4_a1 this[int dim] { get { return Clone<var_u64_v4_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }
        public var_u64 x { get { return Clone<var_u64>(v => v.Type = v.Type.x, v => v.Init = null); } }
        public var_u64 r { get { return Clone<var_u64>(v => v.Type = v.Type.r, v => v.Init = null); } }
        public var_u64 y { get { return Clone<var_u64>(v => v.Type = v.Type.y, v => v.Init = null); } }
        public var_u64 g { get { return Clone<var_u64>(v => v.Type = v.Type.g, v => v.Init = null); } }
        public var_u64 z { get { return Clone<var_u64>(v => v.Type = v.Type.z, v => v.Init = null); } }
        public var_u64 b { get { return Clone<var_u64>(v => v.Type = v.Type.b, v => v.Init = null); } }
        public var_u64 w { get { return Clone<var_u64>(v => v.Type = v.Type.w, v => v.Init = null); } }
        public var_u64 a { get { return Clone<var_u64>(v => v.Type = v.Type.a, v => v.Init = null); } }
        public new var_u64_v4 reg { get { return Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public new var_u64_v4 sreg { get { return Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public new var_u64_v4 local { get { return Clone(v => v.Space = Common.Enumerations.space.local); } }
        public new var_u64_v4 shared { get { return Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public new var_u64_v4 global { get { return Clone(v => v.Space = Common.Enumerations.space.global); } }
        public new var_u64_v4 param { get { return Clone(v => v.Space = Common.Enumerations.space.param); } }
        public new var_u64_v4 const0 { get { return Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public new var_u64_v4 const1 { get { return Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public new var_u64_v4 const2 { get { return Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public new var_u64_v4 const3 { get { return Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public new var_u64_v4 const4 { get { return Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public new var_u64_v4 const5 { get { return Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public new var_u64_v4 const6 { get { return Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public new var_u64_v4 const7 { get { return Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public new var_u64_v4 const8 { get { return Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public new var_u64_v4 const9 { get { return Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public new var_u64_v4 const10 { get { return Clone(v => v.Space = Common.Enumerations.space.const10); } }

        public var_u64_v4 init(ulong4 value) { return Clone(v => v.Init = value); }
        public var_u64_v4 init(ulong3 value) { return Clone(v => v.Init = value); }

        public var_u64_v4() { Alignment = 32 /* sizeof(ulong4) */; }
        public var_u64_v4 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_u64_v4 align32{ get { return align(32); } }
        public var_u64_v4 align64{ get { return align(64); } }
        public var_u64_v4 align128{ get { return align(128); } }
        public var_u64_v4 align256{ get { return align(256); } }

        public var_u64_v4 export { get { return Clone(v => v.IsVisible = true); } }
        public var_u64_v4 import { get { return Clone(v => v.IsExtern = true); } }
        public var_u64_v4 @extern { get { return Clone(v => v.IsExtern = true); } }

        internal var_u64_v4 Clone()
        {
            return Clone<var_u64_v4>();
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

        internal var_u64_v4 Clone(params Action<var_u64_v4>[] mods)
        {
            return Clone<var_u64_v4>(mods);
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
