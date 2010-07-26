using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quanta
{
    [DebuggerNonUserCode]
    public abstract class QuantumAttribute : ParticleAttribute
    {
        protected QuantumAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}