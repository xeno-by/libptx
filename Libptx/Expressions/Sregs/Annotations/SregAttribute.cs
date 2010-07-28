using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Common.Annotations.Atoms;

namespace Libptx.Expressions.Sregs.Annotations
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
    [DebuggerNonUserCode]
    public class SregAttribute : AtomAttribute
    {
        public Type Type { get; set; }

        public SregAttribute(Type type)
            : this(null, type, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public SregAttribute(Type type, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public SregAttribute(Type type, HardwareIsa hardwareIsa)
            : this(null, type, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public SregAttribute(Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : this(null, type, softwareIsa, hardwareIsa)
        {
        }

        public SregAttribute(Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : this(null, type, softwareIsa, hardwareIsa)
        {
        }

        public SregAttribute(String signature, Type type)
            : this(signature, type, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        public SregAttribute(String signature, Type type, SoftwareIsa softwareIsa)
            : this(signature, type, softwareIsa, softwareIsa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        public SregAttribute(String signature, Type type, HardwareIsa hardwareIsa)
            : this(signature, type, hardwareIsa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hardwareIsa)
        {
        }

        public SregAttribute(String signature, Type type, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
            Type = type;
        }

        public SregAttribute(String signature, Type type, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
            Type = type;
        }
    }
}