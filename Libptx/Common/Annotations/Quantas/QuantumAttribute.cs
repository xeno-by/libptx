using System;
using Libcuda.Versions;

namespace Libptx.Common.Annotations.Quantas
{
    internal abstract class QuantumAttribute : ParticleAttribute
    {
        protected QuantumAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
            : base(signature, softwareIsa, hardwareIsa)
        {
        }
    }
}