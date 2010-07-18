using System;
using Libptx.Expressions;
using XenoGears.Assertions;

namespace Libptx.Edsl.Vars
{
    public class Var_U16 : Var
    {
        public Var_U16_V1 v1 { get { return Clone<Var_U16_V1>(v => v.Type = v.Type.v1, v => v.Init = null); } }
        public Var_U16_V2 v2 { get { return Clone<Var_U16_V2>(v => v.Type = v.Type.v2, v => v.Init = null); } }
        public Var_U16_V4 v4 { get { return Clone<Var_U16_V4>(v => v.Type = v.Type.v4, v => v.Init = null); } }
        public Var_U16_A1 this[int dim] { get { return Clone<Var_U16_A1>(v => v.Type = v.Type[dim], v => v.Init = null); } }

        public new Var_U16 reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new Var_U16 sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new Var_U16 local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new Var_U16 shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new Var_U16 global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new Var_U16 param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new Var_U16 @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new Var_U16 const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new Var_U16 const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new Var_U16 const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new Var_U16 const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new Var_U16 const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new Var_U16 const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new Var_U16 const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new Var_U16 const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new Var_U16 const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new Var_U16 const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new Var_U16 const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public Var_U16 init(short value) { return Clone(v => v.Init = value); }

        public Var_U16() { Alignment = sizeof(short); }
        public Var_U16 align(int alignment) { return Clone(v => v.Alignment = alignment.AssertThat(i => i > 0 && i % sizeof(short) == 0)); }

        public Var_U16 export { get { return Clone(v => v.IsVisible = true); } }
        public Var_U16 import { get { return Clone(v => v.IsExtern = true); } }
        public Var_U16 @extern { get { return Clone(v => v.IsExtern = true); } }

        private Var_U16 Clone()
        {
            return Clone<Var_U16>();
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

        protected Var_U16 Clone(params Action<Var_U16>[] mods)
        {
            return Clone<Var_U16>(mods);
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