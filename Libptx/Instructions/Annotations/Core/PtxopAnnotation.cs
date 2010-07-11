using System;
using Libcuda.Versions;

namespace Libptx.Instructions.Annotations.Core
{
    internal abstract class PtxopAnnotation : Attribute
    {
        public ptxopspec spec { get; set; }
        public SoftwareIsa swisa { get; set; }
        public HardwareIsa hwisa { get; set; }

        protected PtxopAnnotation()
            : this(null, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        protected PtxopAnnotation(SoftwareIsa swisa)
            : this(null, swisa, swisa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        protected PtxopAnnotation(HardwareIsa hwisa)
            : this(null, hwisa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hwisa)
        {
        }

        protected PtxopAnnotation(SoftwareIsa swisa, HardwareIsa hwisa)
            : this(null, swisa, hwisa)
        {
        }

        protected PtxopAnnotation(String spec)
            : this(spec, SoftwareIsa.PTX_10, HardwareIsa.SM_10)
        {
        }

        protected PtxopAnnotation(String spec, SoftwareIsa swisa)
            : this(spec, swisa, swisa < SoftwareIsa.PTX_20 ? HardwareIsa.SM_10 : HardwareIsa.SM_20)
        {
        }

        protected PtxopAnnotation(String spec, HardwareIsa hwisa)
            : this(spec, hwisa < HardwareIsa.SM_20 ? SoftwareIsa.PTX_10 : SoftwareIsa.PTX_20, hwisa)
        {
        }

        protected PtxopAnnotation(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
        {
            this.spec = spec;
            this.swisa = swisa;
            this.hwisa = hwisa;
        }
    }
}