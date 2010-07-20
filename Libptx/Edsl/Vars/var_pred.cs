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
    public class var_pred : has_type_pred
    {
        public static var_pred operator !(var_pred var_pred) { return var_pred.Clone(v => v.Mod |= VarMod.Not); }
        public static var_couple operator |(var_pred var_pred1, var_pred var_pred2) { return new var_couple{fst = var_pred1, snd = var_pred2}; }

        public new var_pred reg { get { return Clone(v => v.Space = Common.Enumerations.space.reg); } }
        public new var_pred sreg { get { return Clone(v => v.Space = Common.Enumerations.space.sreg); } }
        public new var_pred local { get { return Clone(v => v.Space = Common.Enumerations.space.local); } }
        public new var_pred shared { get { return Clone(v => v.Space = Common.Enumerations.space.shared); } }
        public new var_pred global { get { return Clone(v => v.Space = Common.Enumerations.space.global); } }
        public new var_pred param { get { return Clone(v => v.Space = Common.Enumerations.space.param); } }
        public new var_pred const0 { get { return Clone(v => v.Space = Common.Enumerations.space.const0); } }
        public new var_pred const1 { get { return Clone(v => v.Space = Common.Enumerations.space.const1); } }
        public new var_pred const2 { get { return Clone(v => v.Space = Common.Enumerations.space.const2); } }
        public new var_pred const3 { get { return Clone(v => v.Space = Common.Enumerations.space.const3); } }
        public new var_pred const4 { get { return Clone(v => v.Space = Common.Enumerations.space.const4); } }
        public new var_pred const5 { get { return Clone(v => v.Space = Common.Enumerations.space.const5); } }
        public new var_pred const6 { get { return Clone(v => v.Space = Common.Enumerations.space.const6); } }
        public new var_pred const7 { get { return Clone(v => v.Space = Common.Enumerations.space.const7); } }
        public new var_pred const8 { get { return Clone(v => v.Space = Common.Enumerations.space.const8); } }
        public new var_pred const9 { get { return Clone(v => v.Space = Common.Enumerations.space.const9); } }
        public new var_pred const10 { get { return Clone(v => v.Space = Common.Enumerations.space.const10); } }

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

        internal var_pred Clone()
        {
            return Clone<var_pred>();
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

        internal var_pred Clone(params Action<var_pred>[] mods)
        {
            return Clone<var_pred>(mods);
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
