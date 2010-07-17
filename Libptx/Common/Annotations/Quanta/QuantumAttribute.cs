using System;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quanta
{
    internal abstract class QuantumAttribute : ParticleAttribute
    {
        protected QuantumAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}