using System;
using System.Diagnostics;
using System.Linq;
using Libptx.Common.Types;
using Libptx.Instructions;
using Libptx.Reflection;
using NUnit.Framework;
using XenoGears.Functional;
using XenoGears.Logging;
using XenoGears.Reflection;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Generics;
using XenoGears.Strings;
using Type = Libptx.Common.Types.Type;
using XenoGears.Assertions;

namespace Libptx.Playground
{
    [TestFixture]
    public class MetaTests
    {
        [Test]
        public void EnsureMostStuffIsMarkedWithDebuggerNonUserCode()
        {
            var asm = AppDomain.CurrentDomain.GetAssemblies().SingleOrDefault(asm1 => asm1.GetName().Name == "Libptx");
            if (asm == null) asm = AppDomain.CurrentDomain.Load("Libptx");

            var types = asm.GetTypes().Where(t => !t.IsInterface).ToReadOnly();
            var failed_types = types
                .Where(t => !t.HasAttr<DebuggerNonUserCodeAttribute>())
                .Where(t => !t.IsCompilerGenerated())
                .Where(t => !t.Name.Contains("<>"))
                .Where(t => !t.Name.Contains("__StaticArrayInit"))
                .Where(t => !t.IsEnum)
                .Where(t => !t.IsDelegate())
                // exceptions for meaty logic
                .ToReadOnly();

            if (failed_types.IsNotEmpty())
            {
                Log.WriteLine(String.Format("{0} types in Libptx aren't marked with [DebuggerNonUserCode]:", failed_types.Count()));
                var messages = failed_types.Select(t => t.GetCSharpRef(ToCSharpOptions.InformativeWithNamespaces));
                messages.OrderDescending().ForEach(message => Log.WriteLine(message));
                Assert.Fail();
            }
        }

        [Test]
        public void EnsurePtxopMetadataIntegrity()
        {
            var libptx = typeof(Module).Assembly;
            var ops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();

            // assert-test #1: all ptxop annotations for given ptxop have the same version and target
            var weirdos = ops.Where(op => op.Particles().Select(pcl => Tuple.New(pcl.Version, pcl.Target)).Distinct().Count() > 1);
            weirdos.ForEach(Console.WriteLine);
            weirdos.AssertEmpty();

            // assert-test #2: all type annotations are mandatory
            ops.SelectMany(op => op.Signatures()).Where(sig => sig.Match(@"\{\.(\w)*type").Success).ForEach(sig => Console.WriteLine(sig));
            var types = Enum.GetValues(typeof(TypeName)).Cast<TypeName>().Select(tn => (Type)tn).Select(t => t.ToString()).ToReadOnly();
            types.ForEach(t => ops.SelectMany(op => op.Signatures()).Where(sig => sig.Match(@"\{\." + t).Success).ForEach(sig => Console.WriteLine(sig)));

            // assert-test #3: verify that ptxop sigs get parsed correctly
            var sigs = ops.ToDictionary(op => op, op => op.PtxopSigs());
            throw new NotImplementedException();
        }

        [Test]
        public void EnsurePtxopValidationCorrectness()
        {
            // todo. implement the following:
            // 1) compose all possible valid combos of ptxop values and compare them with all possible valid combos of ptxop signatures
            // 2) check all possible combos of ops: verify that all valid get compiled by ptxas and all invalid do not

            throw new NotImplementedException();
        }
    }
}