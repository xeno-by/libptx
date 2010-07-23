using System;
using System.IO;
using System.Text;
using Libptx.Expressions;
using Libptx.Expressions.Specials.Annotations;
using XenoGears.Functional;
using XenoGears.Strings;
using System.Linq;
using Libptx.Edsl.TextGenerators.Common;
using XenoGears.Reflection.Attributes;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class SpecialGenerator
    {
        public static void DoGenerate()
        {
            var edsl_base = @"..\..\..\..\Libptx.Edsl\Libptx.Edsl\";
            var dir_specials = edsl_base + @"Expressions\Specials\";
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\..\Libptx.Edsl\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            var libptx = typeof(Special).Assembly;
            var specials = libptx.GetTypes().Where(t => t.BaseType == typeof(Special)).ToReadOnly();
            foreach (var t in specials)
            {
                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.WriteLine("using Libptx.Edsl.Common.Types.Scalar;");
                w.WriteLine("using Libptx.Edsl.Common.Types.Vector;");
                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("namespace {0}", dir2ns(dir_specials));
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public class {0} : {1}, special", t.Name, t.FullName);
                w.WriteLine("{");
                w.Indent++;

                var type = t.Attr<SpecialAttribute>().Type;
                w.EmitTypeSpec(t.Name, type, Space.Other);

                // todo. also emit same stuff as gets emitted for non-reg vars of appropriate type
                // e.g. .x, .y, .z accessors for grid-related special registers

                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");

                var fname = dir_specials + t.Name + ".cs";
                File.WriteAllText(fname, buf.ToString());
            }
        }
    }
}