using System;
using System.Linq;
using Libptx.Expressions;
using Libptx.Common.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_sampler : Var
    {
        public new var_sampler reg { get { return Clone(v => v.Space = Common.Enumerations.Space.Register); } }
        public new var_sampler sreg { get { return Clone(v => v.Space = Common.Enumerations.Space.Special); } }
        public new var_sampler local { get { return Clone(v => v.Space = Common.Enumerations.Space.Local); } }
        public new var_sampler shared { get { return Clone(v => v.Space = Common.Enumerations.Space.Shared); } }
        public new var_sampler global { get { return Clone(v => v.Space = Common.Enumerations.Space.Global); } }
        public new var_sampler param { get { return Clone(v => v.Space = Common.Enumerations.Space.Param); } }
        public new var_sampler @const { get { return Clone(v => v.Space = Common.Enumerations.Space.Const); } }
        public new var_sampler const0 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const0); } }
        public new var_sampler const1 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const1); } }
        public new var_sampler const2 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const2); } }
        public new var_sampler const3 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const3); } }
        public new var_sampler const4 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const4); } }
        public new var_sampler const5 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const5); } }
        public new var_sampler const6 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const6); } }
        public new var_sampler const7 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const7); } }
        public new var_sampler const8 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const8); } }
        public new var_sampler const9 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const9); } }
        public new var_sampler const10 { get { return Clone(v => v.Space = Common.Enumerations.Space.Const10); } }

        public var_sampler init(Sampler value) { return Clone(v => v.Init = value); }

        public var_sampler() { Alignment = 1 /* sizeof(Sampler) */; }
        public var_sampler align(int alignment){ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }
        public var_sampler align1{ get { return align(1); } }
        public var_sampler align2{ get { return align(2); } }
        public var_sampler align4{ get { return align(4); } }
        public var_sampler align8{ get { return align(8); } }

        public var_sampler export { get { return Clone(v => v.IsVisible = true); } }
        public var_sampler import { get { return Clone(v => v.IsExtern = true); } }
        public var_sampler @extern { get { return Clone(v => v.IsExtern = true); } }

        private var_sampler Clone()
        {
            return Clone<var_sampler>();
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

        protected var_sampler Clone(params Action<var_sampler>[] mods)
        {
            return Clone<var_sampler>(mods);
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
