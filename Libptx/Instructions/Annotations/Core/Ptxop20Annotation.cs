using System;
using System.Diagnostics;
using Libcuda.Versions;
using XenoGears.Assertions;

namespace Libptx.Instructions.Annotations.Core
{
    [AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
    [DebuggerNonUserCode]
    internal class Ptxop20Annotation : PtxopAnnotation
    {
        protected Ptxop20Annotation()
            : this(null, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        protected Ptxop20Annotation(SoftwareIsa swisa)
            : this(null, swisa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_20)
        {
        }

        protected Ptxop20Annotation(HardwareIsa hwisa)
            : this(null, SoftwareIsa.PTX_20, hwisa)
        {
        }

        protected Ptxop20Annotation(SoftwareIsa swisa, HardwareIsa hwisa)
            : this(null, swisa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), hwisa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        protected Ptxop20Annotation(String spec)
            : this(spec, SoftwareIsa.PTX_20, HardwareIsa.SM_20)
        {
        }

        protected Ptxop20Annotation(String spec, SoftwareIsa swisa)
            : this(spec, swisa.AssertThat(isa => isa >= SoftwareIsa.PTX_20), HardwareIsa.SM_10)
        {
        }

        protected Ptxop20Annotation(String spec, HardwareIsa hwisa)
            : this(spec, SoftwareIsa.PTX_20, hwisa.AssertThat(isa => isa >= HardwareIsa.SM_20))
        {
        }

        protected Ptxop20Annotation(String spec, SoftwareIsa swisa, HardwareIsa hwisa)
        {
            this.spec = spec;
            this.swisa = swisa.AssertThat(isa => isa >= SoftwareIsa.PTX_20);
            this.hwisa = hwisa.AssertThat(isa => isa >= HardwareIsa.SM_20);
        }
    }
}