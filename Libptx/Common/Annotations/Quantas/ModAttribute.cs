using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quantas
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    internal class ModAttribute : QuantumAttribute
    {
        public ModAttribute()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public ModAttribute(SoftwareIsa softwareIsa)
            : this(null, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public ModAttribute(HardwareIsa hardwareIsa)
            : this(null, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public ModAttribute(SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, softwareIsa, hardwareIsa)
        {
        }

        public ModAttribute(String signature)
            : this(signature, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public ModAttribute(String signature, SoftwareIsa softwareIsa)
            : this(signature, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public ModAttribute(String signature, HardwareIsa hardwareIsa)
            : this(signature, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public ModAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}