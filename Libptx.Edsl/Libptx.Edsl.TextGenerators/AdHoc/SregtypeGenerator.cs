using System;
using System.IO;
using System.Text;
using Libptx.Expressions.Sregs;
using Libptx.Reflection;
using XenoGears.Strings;
using XenoGears.Functional;
using XenoGears.Strings.Writers;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class SregtypeGenerator
    {
        public static void DoGenerate()
        {
            var libptx_base = @"..\..\..\..\Libptx\";
            var libptx = typeof(Module).Assembly;

            var dir_sregs = libptx_base + @"Expressions\Sregs\";
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\..\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            var buf = new StringBuilder();
            var w = new StringWriter(buf).Indented();

            w.WriteLine("namespace {0}", typeof(sregtype).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public enum {0}", typeof(sregtype).Name);
            w.WriteLine("{");
            w.Indent++;
            Sregs.All.ForEach((op, i) => w.WriteLine(op.Name + (i == 0 ? " = 1" : "") + ","));
            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            w.WriteLine("");
            w.WriteLine("namespace {0}", typeof(Sreg).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract partial class {0}", typeof(Sreg).Name);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract sregtype discr { get; }");
            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            Sregs.All.ForEach(op =>
            {
                w.WriteLine("");
                w.WriteLine("namespace {0}", op.Namespace);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0}", op.Name);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public override sregtype discr { get { return sregtype." + op.Name + "; } }");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
            });

            var type_fname = dir_sregs + "sregtype.cs";
            File.WriteAllText(type_fname, buf.ToString());
        }
    }
}