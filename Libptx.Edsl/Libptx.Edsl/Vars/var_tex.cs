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
    public class var_tex : has_type_tex
    {
        public new var_tex reg { get { return Clone(v => v.Space = space.reg); } }
        public new var_tex sreg { get { return Clone(v => v.Space = space.sreg); } }
        public new var_tex local { get { return Clone(v => v.Space = space.local); } }
        public new var_tex shared { get { return Clone(v => v.Space = space.shared); } }
        public new var_tex global { get { return Clone(v => v.Space = space.global); } }
        public new var_tex param { get { return Clone(v => v.Space = space.param); } }
        public new var_tex const0 { get { return Clone(v => v.Space = space.const0); } }
        public new var_tex const1 { get { return Clone(v => v.Space = space.const1); } }
        public new var_tex const2 { get { return Clone(v => v.Space = space.const2); } }
        public new var_tex const3 { get { return Clone(v => v.Space = space.const3); } }
        public new var_tex const4 { get { return Clone(v => v.Space = space.const4); } }
        public new var_tex const5 { get { return Clone(v => v.Space = space.const5); } }
        public new var_tex const6 { get { return Clone(v => v.Space = space.const6); } }
        public new var_tex const7 { get { return Clone(v => v.Space = space.const7); } }
        public new var_tex const8 { get { return Clone(v => v.Space = space.const8); } }
        public new var_tex const9 { get { return Clone(v => v.Space = space.const9); } }
        public new var_tex const10 { get { return Clone(v => v.Space = space.const10); } }

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

        internal var_tex Clone()
        {
            return Clone<var_tex>();
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

        internal var_tex Clone(params Action<var_tex>[] mods)
        {
            return Clone<var_tex>(mods);
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