using System;
using System.Collections.Generic;
using System.IO;
using Libcuda.Versions;
using Libptx.Common.Enumerations;
using Libptx.Common.Annotations.Atoms;
using Libptx.Common.Types;
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

        protected static Type u8 { get { return TypeName.U8; } }
        protected static Type s8 { get { return TypeName.S8; } }
        protected static Type u16 { get { return TypeName.U16; } }
        protected static Type s16 { get { return TypeName.S16; } }
        protected static Type u32 { get { return TypeName.U32; } }
        protected static Type s32 { get { return TypeName.S32; } }
        protected static Type u64 { get { return TypeName.U64; } }
        protected static Type s64 { get { return TypeName.S64; } }
        protected static Type f16 { get { return TypeName.F16; } }
        protected static Type f32 { get { return TypeName.F32; } }
        protected static Type f64 { get { return TypeName.F64; } }
        protected static Type b8 { get { return TypeName.B8; } }
        protected static Type b16 { get { return TypeName.B16; } }
        protected static Type b32 { get { return TypeName.B32; } }
        protected static Type b64 { get { return TypeName.B64; } }
        protected static Type pred { get { return TypeName.Pred; } }
        protected static Type tex { get { return TypeName.Tex; } }
        protected static Type sampler { get { return TypeName.Sampler; } }
        protected static Type surf { get { return TypeName.Surf; } }

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