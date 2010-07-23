using System;
using System.IO;
using System.Linq;
using System.Text;
using Libptx.Expressions;
using Libptx.Expressions.Specials;
using XenoGears.Strings;
using XenoGears.Functional;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class SpecialtypeGenerator
    {
        public static void DoGenerate()
        {
            var libptx_base = @"..\..\..\..\Libptx\";
            var libptx = typeof(Module).Assembly;
            var ops = libptx.GetTypes().Where(t => t.Namespace == typeof(tid).Namespace)
                .Where(t => t != typeof(specialtype))
                .OrderBy(t => t.Name);

            var dir_specials = libptx_base + @"Expressions\Specials\";
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\..\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            var buf = new StringBuilder();
            var w = new StringWriter(buf).Indented();
            w.WriteLine("using {0};", typeof(specialtype).Namespace);
            w.WriteLineNoTabs(String.Empty);

            w.WriteLine("namespace {0}", typeof(specialtype).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public enum {0}", typeof(specialtype).Name);
            w.WriteLine("{");
            w.Indent++;
            ops.ForEach((op, i) => w.WriteLine(op.Name + (i == 0 ? " = 1" : "") + ","));
            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            w.WriteLine("");
            w.WriteLine("namespace {0}", typeof(Special).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract partial class {0}", typeof(Special).Name);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract specialtype discr { get; }");
            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            ops.ForEach(op =>
            {
                w.WriteLine("");
                w.WriteLine("namespace {0}", op.Namespace);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0}", op.Name);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public override specialtype discr { get { return specialtype." + op.Name + "; } }");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
            });

            var type_fname = dir_specials + "specialtype.cs";
            File.WriteAllText(type_fname, buf.ToString());
        }
    }
}