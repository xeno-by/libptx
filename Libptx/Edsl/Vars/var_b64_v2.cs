using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_b64_v2 : var
    {
        public var_b64_v2_a1 this[int dim] { get { return Clone<var_b64_v2_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new var_b64_v2 reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_b64_v2 sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_b64_v2 local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_b64_v2 shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_b64_v2 global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_b64_v2 param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_b64_v2 @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_b64_v2 const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_b64_v2 const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_b64_v2 const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_b64_v2 const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_b64_v2 const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_b64_v2 const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_b64_v2 const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_b64_v2 const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_b64_v2 const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_b64_v2 const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_b64_v2 const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_b64_v2 init(Bit64_V2 value) { return Clone(v => v.Init = value); }

        public var_b64_v2() { Alignment = 16 /* sizeof(Bit64_V2) */; }
        public var_b64_v2 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_b64_v2 align16{ get { return align(16); } }
        public var_b64_v2 align32{ get { return align(32); } }
        public var_b64_v2 align64{ get { return align(64); } }
        public var_b64_v2 align128{ get { return align(128); } }

        public var_b64_v2 export { get { return Clone(v => v.IsVisible = true); } }
        public var_b64_v2 import { get { return Clone(v => v.IsExtern = true); } }
        public var_b64_v2 @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_b64_v2 Clone()
        {
            return Clone<var_b64_v2>();
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

        protected var_b64_v2 Clone(params Action<var_b64_v2>[] mods)
        {
            return Clone<var_b64_v2>(mods);
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
