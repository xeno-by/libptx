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

                // uncommenting and running this will kill all the customizations!
//                PtxopGenerator.DoGenerate();

                TypeGenerator.DoGenerate();
                SpecialGenerator.DoGenerate();
                VectorGenerator.DoGenerate();
            }
        }
    }
}