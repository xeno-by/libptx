using System;
using System.Linq;
using Libptx.Expressions;
using Libptx.Common.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_pred : Var
    {
        public new var_pred reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_pred sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_pred local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_pred shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_pred global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_pred param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_pred @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_pred const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_pred const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_pred const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_pred const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_pred const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_pred const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_pred const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_pred const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_pred const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_pred const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_pred const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_pred init(bool value) { return Clone(v => v.Init = value); }

        public var_pred() { Alignment = 4 /* sizeof(bool) */; }
        public var_pred align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_pred align4{ get { return align(4); } }
        public var_pred align8{ get { return align(8); } }
        public var_pred align16{ get { return align(16); } }
        public var_pred align32{ get { return align(32); } }

        public var_pred export { get { return Clone(v => v.IsVisible = true); } }
        public var_pred import { get { return Clone(v => v.IsExtern = true); } }
        public var_pred @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_pred Clone()
        {
            return Clone<var_pred>();
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

        protected var_pred Clone(params Action<var_pred>[] mods)
        {
            return Clone<var_pred>(mods);
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