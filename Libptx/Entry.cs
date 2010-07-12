using System.Runtime.InteropServices;
using Libcuda.Versions;
using XenoGears.Assertions;
using System.Linq;

namespace Libptx
{
    public class Entry : Func
    {
        public Tuning Tuning { get; set; }
        public Entry() { Tuning = new Tuning(); }

        // todo. implement this:
        // For PTX ISA version 1.4 and later, parameter variables are declared in the kernel
        // parameter list. For PTX ISA versions 1.0 through 1.3, parameter variables are
        // declared in the kernel body.

        public override void Validate()
        {
            Params.AssertEach(p => p.Space == param);
            Rets.AssertEmpty();

            var size_limit = 256;
            if (Ctx.Version >= SoftwareIsa.PTX_15) size_limit += 4096;
            (Params.Sum(p => Marshal.SizeOf(p.Type)) <= size_limit).AssertTrue();

            Tuning.Validate();
        }
    }
}