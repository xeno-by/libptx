using System;
using System.Collections.Generic;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Instructions;
using XenoGears.Functional;
using Libptx.Common.Annotations;
using XenoGears.Strings;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Playground
{
    internal class TestRunner
    {
        public static void Main(String[] args)
        {
            var libptx = typeof(Module).Assembly;
            var ops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();

            // assert-test #1: all ptxop annotations for given ptxop have the same version and target
            var weirdos = ops.Where(op => op.Particles().Select(pcl => Tuple.New(pcl.Version, pcl.Target)).Distinct().Count() > 1);
            weirdos.ForEach(Console.WriteLine);

            // assert-test #2: all type annotations are mandatory
            ops.SelectMany(op => op.Signatures()).Where(sig => sig.Match(@"\{\.(\w)*type").Success).ForEach(sig => Console.WriteLine(sig));
            var types = Enum.GetValues(typeof(TypeName)).Cast<TypeName>().Select(tn => (Type)tn).Select(t => t.ToString()).ToReadOnly();
            types.ForEach(t => ops.SelectMany(op => op.Signatures()).Where(sig => sig.Match(@"\{\." + t).Success).ForEach(sig => Console.WriteLine(sig)));

//            // see more details at http://www.nunit.org/index.php?p=consoleCommandLine&r=2.5.5
//            var nunitArgs = new List<String>();
//            nunitArgs.Add("/run:Libptx.Playground");
//            nunitArgs.Add("/include:Hot");
//            nunitArgs.Add("/domain:None");
//            nunitArgs.Add("/noshadow");
//            nunitArgs.Add("/nologo");
//            nunitArgs.Add("Libptx.Playground.exe");
//            NUnit.ConsoleRunner.Runner.Main(nunitArgs.ToArray());
        }
    }
}
