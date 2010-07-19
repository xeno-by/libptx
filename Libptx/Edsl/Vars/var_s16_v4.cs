using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_s16_v4 : var
    {
        public var_s16_v4_a1 this[int dim] { get { return Clone<var_s16_v4_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new var_s16_v4 reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_s16_v4 sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_s16_v4 local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_s16_v4 shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_s16_v4 global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_s16_v4 param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_s16_v4 @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_s16_v4 const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_s16_v4 const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_s16_v4 const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_s16_v4 const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_s16_v4 const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_s16_v4 const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_s16_v4 const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_s16_v4 const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_s16_v4 const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_s16_v4 const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_s16_v4 const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_s16_v4 init(short4 value) { return Clone(v => v.Init = value); }
        public var_s16_v4 init(short3 value) { return Clone(v => v.Init = value); }

        public var_s16_v4() { Alignment = 8 /* sizeof(short4) */; }
        public var_s16_v4 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_s16_v4 align8{ get { return align(8); } }
        public var_s16_v4 align16{ get { return align(16); } }
        public var_s16_v4 align32{ get { return align(32); } }
        public var_s16_v4 align64{ get { return align(64); } }

        public var_s16_v4 export { get { return Clone(v => v.IsVisible = true); } }
        public var_s16_v4 import { get { return Clone(v => v.IsExtern = true); } }
        public var_s16_v4 @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_s16_v4 Clone()
        {
            return Clone<var_s16_v4>();
        }

        private T Clone<T>()
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

        protected var_s16_v4 Clone(params Action<var_s16_v4>[] mods)
        {
            return Clone<var_s16_v4>(mods);
        }

        protected T Clone<T>(params Action<T>[] mods)
            where T : var, new()
        {
            T clone = Clone<T>();
            foreach (Action<T> mod in mods) mod(clone);
            return clone;
        }
    }
}
