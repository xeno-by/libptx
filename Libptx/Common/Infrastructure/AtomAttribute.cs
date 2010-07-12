using System;
using Libcuda.Versions;

namespace Libptx.Common.Infrastructure
{
    public abstract class AtomAttribute : Attribute
    {
        public virtual String Signature { get; set; }
        public virtual SoftwareIsa Version { get; set; }
        public virtual HardwareIsa Target { get; set; }

        protected AtomAttribute(String signature, SoftwareIsa softwareIsa, HardwareIsa hardwareIsa)
        {
            Signature = signature;
            Version = softwareIsa;
            Target = hardwareIsa;
        }
    }
}