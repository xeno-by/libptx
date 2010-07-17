using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quantas
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class AffixAttribute : QuantumAttribute
    {
        public AffixAttribute()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public AffixAttribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public AffixAttribute(HardwareIsa hardwareIsa)
            : this(null, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public AffixAttribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public AffixAttribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public AffixAttribute(String signature)
            : this(signature, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public AffixAttribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public AffixAttribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public AffixAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }

        public AffixAttribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}