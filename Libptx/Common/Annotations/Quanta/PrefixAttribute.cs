using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quanta
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    public class PrefixAttribute : AffixAttribute
    {
        public PrefixAttribute()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public PrefixAttribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public PrefixAttribute(HardwareIsa hardwareIsa)
            : this(null, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public PrefixAttribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public PrefixAttribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public PrefixAttribute(String signature)
            : this(signature, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public PrefixAttribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public PrefixAttribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public PrefixAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }

        public PrefixAttribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}