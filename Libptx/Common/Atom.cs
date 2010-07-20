using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Atoms;
using Libptx.Common.Types;
using Libptx.Edsl.Types;
using Type=Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Common
{
    public abstract class Atom : Validatable, Renderable
    {
        private IList<Location> _locations = new List<Location>();
        public IList<Location> Locations
        {
            get { return _locations; }
            set { _locations = value ?? new List<Location>(); }
        }

        private IList<String> _pragmas = new List<String>();
        public IList<String> Pragmas
        {
            get { return _pragmas; }
            set { _pragmas = value ?? new List<String>(); }
        }

        public SoftwareIsa Version { get { return (SoftwareIsa)Math.Max((int)CoreVersion, (int)CustomVersion); } }
        protected SoftwareIsa CoreVersion { get { throw new NotImplementedException(); } }
        protected virtual SoftwareIsa CustomVersion { get { return SoftwareIsa.PTX_10; } }

        public HardwareIsa Target { get { return (HardwareIsa)Math.Max((int)CoreTarget, (int)CustomTarget); } }
        protected HardwareIsa CoreTarget { get { throw new NotImplementedException(); } }
        protected virtual HardwareIsa CustomTarget { get { return HardwareIsa.SM_10; } }

        public void Validate(Module ctx) { (ctx.Version >= Version).AssertTrue(); (ctx.Target >= Target).AssertTrue(); CustomValidate(ctx); }
        protected virtual void CustomValidate(Module ctx) { /* do nothing */ }

        void Renderable.RenderAsPtx(TextWriter writer) { RenderAsPtx(writer); }
        protected abstract void RenderAsPtx(TextWriter writer);
        public sealed override String ToString() { return this.RenderAsPtx(); }

        #region Enumeration values => Static properties

        protected static u8 u8 { get { return new u8(); } }
        protected static s8 s8 { get { return new s8(); } }
        protected static u16 u16 { get { return new u16(); } }
        protected static s16 s16 { get { return new s16(); } }
        protected static u32 u32 { get { return new u32(); } }
        protected static s32 s32 { get { return new s32(); } }
        protected static u64 u64 { get { return new u64(); } }
        protected static s64 s64 { get { return new s64(); } }
        protected static f16 f16 { get { return new f16(); } }
        protected static f32 f32 { get { return new f32(); } }
        protected static f64 f64 { get { return new f64(); } }
        protected static b8 b8 { get { return new b8(); } }
        protected static b16 b16 { get { return new b16(); } }
        protected static b32 b32 { get { return new b32(); } }
        protected static b64 b64 { get { return new b64(); } }
        protected static pred pred { get { return new pred(); } }
        protected static tex tex { get { return new tex(); } }
        protected static sampler sampler { get { return new sampler(); } }
        protected static surf surf { get { return new surf(); } }

        protected static cmp eq { get { return cmp.eq; } }
        protected static cmp ne { get { return cmp.ne; } }
        protected static cmp lt { get { return cmp.lt; } }
        protected static cmp le { get { return cmp.le; } }
        protected static cmp gt { get { return cmp.gt; } }
        protected static cmp ge { get { return cmp.ge; } }
        protected static cmp lo { get { return cmp.lo; } }
        protected static cmp ls { get { return cmp.ls; } }
        protected static cmp hi { get { return cmp.hi; } }
        protected static cmp hs { get { return cmp.hs; } }
        protected static cmp equ { get { return cmp.equ; } }
        protected static cmp neu { get { return cmp.neu; } }
        protected static cmp ltu { get { return cmp.ltu; } }
        protected static cmp leu { get { return cmp.leu; } }
        protected static cmp gtu { get { return cmp.gtu; } }
        protected static cmp geu { get { return cmp.geu; } }
        protected static cmp num { get { return cmp.num; } }
        protected static cmp nan { get { return cmp.nan; } }

        protected static space reg { get { return space.reg; } }
        protected static space sreg { get { return space.sreg; } }
        protected static space local { get { return space.local; } }
        protected static space shared { get { return space.shared; } }
        protected static space global { get { return space.global; } }
        protected static space param { get { return space.param; } }
        protected static space @const { get { return space.@const; } }
        protected static space const0 { get { return space.const0; } }
        protected static space const1 { get { return space.const1; } }
        protected static space const2 { get { return space.const2; } }
        protected static space const3 { get { return space.const3; } }
        protected static space const4 { get { return space.const4; } }
        protected static space const5 { get { return space.const5; } }
        protected static space const6 { get { return space.const6; } }
        protected static space const7 { get { return space.const7; } }
        protected static space const8 { get { return space.const8; } }
        protected static space const9 { get { return space.const9; } }
        protected static space const10 { get { return space.const10; } }

        #endregion
    }
}