using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Bindings
{
    // todo. also support the following scenarios:
    // 1) multiple instructions of PTX assembly
    // 2) linkage to pre-compiled binaries stored in native libs, e.g. within CUBLAS or CUFFT
    // 3) linkage to PTX generator method

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    [DebuggerNonUserCode]
    public class PtxAttribute : Attribute
    {
        public String Code { get; private set; }
        public HardwareIsa HardwareIsa { get; set; }
        public SoftwareIsa SoftwareIsa { get; set; }

        public PtxAttribute(String code)
            : this(code, HardwareIsa.SM_10, SoftwareIsa.PTX_10)
        {
        }

        public PtxAttribute(String code, HardwareIsa hardwareIsa)
            : this(code, hardwareIsa, SoftwareIsa.PTX_10)
        {
        }

        public PtxAttribute(String code, HardwareIsa hardwareIsa, SoftwareIsa softwareIsa)
        {
            Code = code;
            HardwareIsa = hardwareIsa;
            SoftwareIsa = softwareIsa;
        }
    }
}