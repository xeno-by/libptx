using System;
using System.Diagnostics;
using Libcuda.Versions;

namespace Libptx.Bindings
{
    // todo. support the following scenarios:
    // 1) single instruction of PTX assembly
    // 2) multiple instructions of PTX assembly
    // 3) linkage to PTX generator method
    // 4) linkage to embedded text in resources
    // 5) linkage to pre-compiled binaries stored in native libs, e.g. within CUBLAS or CUFFT

    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    [DebuggerNonUserCode]
    public class PtxAttribute : Attribute
    {
        public String Code { get; private set; }
        public SoftwareIsa Version { get; set; }
        public HardwareIsa Target { get; set; }

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
            Target = hardwareIsa;
            Version = softwareIsa;
        }
    }
}