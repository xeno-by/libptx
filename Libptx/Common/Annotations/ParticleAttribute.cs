using System;
using Libcuda.Versions;

namespace Libptx.Common.Annotations
{
    internal abstract class ParticleAttribute : Attribute
    {
        public String Signature { get; set; }
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

        protected ParticleAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
        {
            Signature = signature;
            Version = softwareIsa;
            Target = hardwareIsa;
        }
    }
}