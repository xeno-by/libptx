using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quanta
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    public class SuffixAttribute : AffixAttribute
    {
        public SuffixAttribute()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public SuffixAttribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public SuffixAttribute(HardwareIsa hardwareIsa)
            : this(null, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public SuffixAttribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public SuffixAttribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public SuffixAttribute(String signature)
            : this(signature, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public SuffixAttribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public SuffixAttribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public SuffixAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }

        public SuffixAttribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}