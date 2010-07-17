using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quantas
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class InfixAttribute : AffixAttribute
    {
        public InfixAttribute()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public InfixAttribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public InfixAttribute(HardwareIsa hardwareIsa)
            : this(null, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public InfixAttribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public InfixAttribute(HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public InfixAttribute(String signature)
            : this(signature, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public InfixAttribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public InfixAttribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public InfixAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }

        public InfixAttribute(String signature, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}