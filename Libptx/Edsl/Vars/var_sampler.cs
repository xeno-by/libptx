using System;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Edsl.Vars.Types;
using Libcuda.DataTypes;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Edsl.Vars
{
    public class var_sampler : has_type_sampler
    {
        public new var_sampler reg { get { return Clone(v => v.Space = Common.Enumerations.space.Register); } }
        public new var_sampler sreg { get { return Clone(v => v.Space = Common.Enumerations.space.Special); } }
        public new var_sampler local { get { return Clone(v => v.Space = Common.Enumerations.space.Local); } }
        public new var_sampler shared { get { return Clone(v => v.Space = Common.Enumerations.space.Shared); } }
        public new var_sampler global { get { return Clone(v => v.Space = Common.Enumerations.space.Global); } }
        public new var_sampler param { get { return Clone(v => v.Space = Common.Enumerations.space.Param); } }
        public new var_sampler @const { get { return Clone(v => v.Space = Common.Enumerations.space.Const); } }
        public new var_sampler const0 { get { return Clone(v => v.Space = Common.Enumerations.space.Const0); } }
        public new var_sampler const1 { get { return Clone(v => v.Space = Common.Enumerations.space.Const1); } }
        public new var_sampler const2 { get { return Clone(v => v.Space = Common.Enumerations.space.Const2); } }
        public new var_sampler const3 { get { return Clone(v => v.Space = Common.Enumerations.space.Const3); } }
        public new var_sampler const4 { get { return Clone(v => v.Space = Common.Enumerations.space.Const4); } }
        public new var_sampler const5 { get { return Clone(v => v.Space = Common.Enumerations.space.Const5); } }
        public new var_sampler const6 { get { return Clone(v => v.Space = Common.Enumerations.space.Const6); } }
        public new var_sampler const7 { get { return Clone(v => v.Space = Common.Enumerations.space.Const7); } }
        public new var_sampler const8 { get { return Clone(v => v.Space = Common.Enumerations.space.Const8); } }
        public new var_sampler const9 { get { return Clone(v => v.Space = Common.Enumerations.space.Const9); } }
        public new var_sampler const10 { get { return Clone(v => v.Space = Common.Enumerations.space.Const10); } }

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

        internal var_sampler Clone()
        {
            return Clone<var_sampler>();
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

        internal var_sampler Clone(params Action<var_sampler>[] mods)
        {
            return Clone<var_sampler>(mods);
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
