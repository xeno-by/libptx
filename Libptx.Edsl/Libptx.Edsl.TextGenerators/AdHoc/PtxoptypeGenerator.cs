using System;
using System.IO;
using System.Linq;
using System.Text;
using Libptx.Instructions;
using Libptx.Instructions.Annotations;
using XenoGears.Strings;
using XenoGears.Functional;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class PtxoptypeGenerator
    {
        public static void DoGenerate()
        {
            var libptx_base = @"..\..\..\..\Libptx\";
            var libptx = typeof(Module).Assembly;
            var ops = libptx.GetTypes().Where(t => t.Namespace.StartsWith(typeof(ptxop).Namespace))
                .Where(t => t.Namespace != typeof(PtxopAttribute).Namespace && t.Namespace != typeof(ptxop).Namespace)
                .OrderBy(t => t.Name);

            var dir_instructions = libptx_base + @"Instructions\";
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\..\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            var buf = new StringBuilder();
            var w = new StringWriter(buf).Indented();
            w.WriteLine("namespace {0}", typeof(ptxoptype).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public enum {0}", typeof(ptxoptype).Name);
            w.WriteLine("{");
            w.Indent++;
            ops.ForEach((op, i) => w.WriteLine(op.Name + (i == 0 ? " = 1" : "") + ","));
            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            w.WriteLine("");
            w.WriteLine("namespace {0}", typeof(ptxop).Namespace);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract partial class {0}", typeof(ptxop).Name);
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public abstract ptxoptype discr { get; }");
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
                w.WriteLine("public override ptxoptype discr { get { return ptxoptype." + op.Name + "; } }");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
            });

            var type_fname = dir_instructions + "ptxoptype.cs";
            File.WriteAllText(type_fname, buf.ToString());
        }
    }
}