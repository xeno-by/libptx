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
                SpecialtypeGenerator.DoGenerate();

                TypeGenerator.DoGenerate();
                SpecialGenerator.DoGenerate();
                VectorGenerator.DoGenerate();

                // todo:
                // 1) compile-time finite automaton for creating ptxops
                // 2) compile-time finite automaton for declaring vars
                // 3) we also need to type constants and provide casts for them as well
                // 4) there should be a way to emit both named and unnamed vars (reg.u32 and reg.u32("foo"))
                // 5) there should be a way to infer var names from the names of locals (debug-mode only, of course)
                // 6) the latter should be optional since it implies serious performance degradation
                // 7) don't forget to emit [DebuggerNonUserCode]
            }
        }
    }
}