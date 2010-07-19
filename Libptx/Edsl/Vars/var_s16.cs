using System;
using System.Linq;
using Libptx.Expressions;
using Libptx.Common.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_s16 : Var
    {
        public var_s16_v1 v1 { get { return Clone<var_s16_v1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public var_s16_v2 v2 { get { return Clone<var_s16_v2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public var_s16_v4 v4 { get { return Clone<var_s16_v4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public var_s16_a1 this[int dim] { get { return Clone<var_s16_a1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new var_s16 reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_s16 sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_s16 local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_s16 shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_s16 global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_s16 param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_s16 @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_s16 const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_s16 const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_s16 const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_s16 const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_s16 const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_s16 const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_s16 const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_s16 const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_s16 const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_s16 const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_s16 const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_s16 init(short value) { return Clone(v => v.Init = value); }

        public var_s16() { Alignment = 2 /* sizeof(short) */; }
        public var_s16 align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_s16 align2{ get { return align(2); } }
        public var_s16 align4{ get { return align(4); } }
        public var_s16 align8{ get { return align(8); } }
        public var_s16 align16{ get { return align(16); } }

        public var_s16 export { get { return Clone(v => v.IsVisible = true); } }
        public var_s16 import { get { return Clone(v => v.IsExtern = true); } }
        public var_s16 @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_s16 Clone()
        {
            return Clone<var_s16>();
        }

        private T Clone<T>()
            where T : Var, new()
        {
            var clone = new T();
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

        protected var_s16 Clone(params Action<var_s16>[] mods)
        {
            return Clone<var_s16>(mods);
        }

        protected T Clone<T>(params Action<T>[] mods)
            where T : Var, new()
        {
            var clone = Clone<T>();
            foreach (var mod in mods) mod(clone);
            return clone;
        }
    }
}
