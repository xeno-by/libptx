using System;
using Libcuda.Versions;
using Libptx.Edsl.TextGenerators.AdHoc;

namespace Libptx.Edsl.TextGenerators
{
    internal class Program
    {
        public static void Main(String[] args)
        {
            using (new Context(SoftwareIsa.PTX_21, HardwareIsa.SM_20))
            {
                PtxoptypeGenerator.DoGenerate();
                SregtypeGenerator.DoGenerate();

                TypeGenerator.DoGenerate();
                ConstGenerator.DoGenerate();
                VectorGenerator.DoGenerate();
                SregGenerator.DoGenerate();

                // todo:
                // 1) compile-time finite automaton for creating ptxops
                // 2) compile-time finite automaton for declaring regs and vars
                // 3) we also need to type constants and provide casts for them as well
                // 4) there should be a way to emit both named and unnamed regs and vars (reg.u32 and reg.u32("foo"))
                // 5) there should be a way to infer reg and var names from the names of locals (debug-mode only, of course)
                // 6) the latter should be optional since it implies serious performance degradation
                // 7) don't forget to emit [DebuggerNonUserCode]
            }
        }
    }
}