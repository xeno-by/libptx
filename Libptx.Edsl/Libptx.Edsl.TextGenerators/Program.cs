using System;
using System.IO;
using System.Linq;
using Libcuda.Versions;
using Libptx.Edsl.TextGenerators.AdHoc;
using Libptx.Instructions;
using XenoGears.Functional;

namespace Libptx.Edsl.TextGenerators
{
    internal class Program
    {
        public static void Main(String[] args)
        {
//            using (new Context(SoftwareIsa.PTX_21, HardwareIsa.SM_20))
//            {
//                PtxoptypeGenerator.DoGenerate();
//                SpecialtypeGenerator.DoGenerate();
//
//                // uncommenting and running this will kill all the customizations!
////                PtxopGenerator.DoGenerate();
//
//                TypeGenerator.DoGenerate();
//                SpecialGenerator.DoGenerate();
//                VectorGenerator.DoGenerate();
//            }

            var libptx = typeof(ptxop).Assembly;
            var ops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();
            foreach (var op in ops)
            {
                var dir = @"..\..\..\..\" + op.Namespace.Replace(".", @"\") + @"\";
                var file = dir + op.Name + ".cs";
                var text = File.ReadAllText(file);
                text = text.Replace(op.Name + "()", "public " + op.Name + "()");
                File.WriteAllText(file, text);
            }
        }
    }
}