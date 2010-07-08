using System;
using System.Diagnostics;
using System.Linq;
using NUnit.Framework;
using XenoGears.Functional;
using XenoGears.Logging;
using XenoGears.Reflection;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Generics;
using XenoGears.Strings;

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
    }
}