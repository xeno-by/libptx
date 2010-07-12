using System;
using System.Collections.Generic;
using Libcuda.Versions;
using Libptx.Common.Enumerations;

namespace Libptx.Common.Infrastructure
{
    public abstract class Atom
    {
        public virtual Module Ctx { get; set; }
        public IList<String> Pragmas { get; set; }

        public SoftwareIsa Version { get { return (SoftwareIsa)Math.Max((int)CoreVersion, (int)CustomVersion); } }
        protected SoftwareIsa CoreVersion { get { throw new NotImplementedException(); } }
        protected virtual SoftwareIsa CustomVersion { get { return SoftwareIsa.PTX_10; } }

        public HardwareIsa Target { get { return (HardwareIsa)Math.Max((int)CoreTarget, (int)CustomTarget); } }
        protected HardwareIsa CoreTarget { get { throw new NotImplementedException(); } }
        protected virtual HardwareIsa CustomTarget { get { return HardwareIsa.SM_10; } }

        public abstract void Validate();
        public String Render() { Validate(); return DoRender(); }
        protected abstract String DoRender();
        public sealed override String ToString() { return Render(); }

        #region Enumeration values => Static properties

        protected static Type u8 { get { return TypeName.U8; } }
        protected static Type s8 { get { return TypeName.S8; } }
        protected static Type u16 { get { return TypeName.U16; } }
        protected static Type s16 { get { return TypeName.S16; } }
        protected static Type u24 { get { return TypeName.U24; } }
        protected static Type s24 { get { return TypeName.S24; } }
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

        protected static Comparison eq { get { return Comparison.Equal; } }
        protected static Comparison ne { get { return Comparison.NotEqual; } }
        protected static Comparison lt { get { return Comparison.LessThan; } }
        protected static Comparison le { get { return Comparison.LessThanOrEqual; } }
        protected static Comparison gt { get { return Comparison.GreaterThan; } }
        protected static Comparison ge { get { return Comparison.GreaterThanOrEqual; } }
        protected static Comparison lo { get { return Comparison.Lower; } }
        protected static Comparison ls { get { return Comparison.LowerOrSame; } }
        protected static Comparison hi { get { return Comparison.Higher; } }
        protected static Comparison hs { get { return Comparison.HigherOrSame; } }
        protected static Comparison equ { get { return Comparison.EqualUnordered; } }
        protected static Comparison neu { get { return Comparison.NotEqualUnordered; } }
        protected static Comparison ltu { get { return Comparison.LessThanUnordered; } }
        protected static Comparison leu { get { return Comparison.LessThanOrEqualUnordered; } }
        protected static Comparison gtu { get { return Comparison.GreaterThanUnordered; } }
        protected static Comparison geu { get { return Comparison.GreaterThanOrEqualUnordered; } }
        protected static Comparison num { get { return Comparison.BothNumbers; } }
        protected static Comparison nan { get { return Comparison.AnyNan; } }

        protected static Space reg { get { return Space.Register; } }
        protected static Space sreg { get { return Space.Special; } }
        protected static Space local { get { return Space.Local; } }
        protected static Space shared { get { return Space.Shared; } }
        protected static Space global { get { return Space.Global; } }
        protected static Space param { get { return Space.Param; } }
        protected static Space @const { get { return Space.Const; } }
        protected static Space const0 { get { return Space.Const0; } }
        protected static Space const1 { get { return Space.Const1; } }
        protected static Space const2 { get { return Space.Const2; } }
        protected static Space const3 { get { return Space.Const3; } }
        protected static Space const4 { get { return Space.Const4; } }
        protected static Space const5 { get { return Space.Const5; } }
        protected static Space const6 { get { return Space.Const6; } }
        protected static Space const7 { get { return Space.Const7; } }
        protected static Space const8 { get { return Space.Const8; } }
        protected static Space const9 { get { return Space.Const9; } }
        protected static Space const10 { get { return Space.Const10; } }

        #endregion
    }
}