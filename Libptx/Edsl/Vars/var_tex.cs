using System;
using System.Linq;
using Libptx.Expressions;
using Libptx.Common.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_tex : Var
    {
        public new var_tex reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_tex sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_tex local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_tex shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_tex global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_tex param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_tex @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_tex const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_tex const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_tex const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_tex const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_tex const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_tex const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_tex const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_tex const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_tex const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_tex const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_tex const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_tex init(Tex value) { return Clone(v => v.Init = value); }

        public var_tex() { Alignment = 1 /* sizeof(Tex) */; }
        public var_tex align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_tex align1{ get { return align(1); } }
        public var_tex align2{ get { return align(2); } }
        public var_tex align4{ get { return align(4); } }
        public var_tex align8{ get { return align(8); } }

        public var_tex export { get { return Clone(v => v.IsVisible = true); } }
        public var_tex import { get { return Clone(v => v.IsExtern = true); } }
        public var_tex @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_tex Clone()
        {
            return Clone<var_tex>();
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

        protected var_tex Clone(params Action<var_tex>[] mods)
        {
            return Clone<var_tex>(mods);
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
