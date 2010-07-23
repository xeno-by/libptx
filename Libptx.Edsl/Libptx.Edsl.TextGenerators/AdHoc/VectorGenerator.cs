using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Libptx.Edsl.TextGenerators.Common;
using Libptx.Expressions;
using XenoGears.Functional;
using XenoGears.Strings;
using Libptx.Common.Types;
using Libptx.Common.Annotations;
using XenoGears.Assertions;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class VectorGenerator
    {
        public static void DoGenerate()
        {
            var edsl_base = @"..\..\..\..\Libptx.Edsl\Libptx.Edsl\";
            var dir_vectors = edsl_base + @"Expressions\Vectors\";
            Func<String, String> dir2ns = dir =>
            {
                var rel = dir.Replace(@"..\..\..\..\Libptx.Edsl\", String.Empty);
                rel = (rel + @"\").Unfold(s => s.Slice(0, -1), s => s.EndsWith(@"\")).Last();
                return rel.Replace(@"\", ".").Slice(0, -1);
            };

            var shortcuts = new Dictionary<int, List<String>>();
            new []{1, 2, 4}.ForEach(i => shortcuts.Add(i, new List<String>()));

            Types.Vector.ForEach(t =>
            {
                t.is_vec().AssertTrue();
                var name = String.Format("v{0}_{1}", t.vec_rank(), t.vec_el().Name.Signature());

                var buf_vec = new StringBuilder();
                var w_vec = new StringWriter(buf_vec).Indented();
                w_vec.WriteLine("using {0};", typeof(AssertionHelper).Namespace);
                w_vec.WriteLine("using {0};", typeof(Libptx.Edsl.Expressions.Vars.var).Namespace);
                w_vec.WriteLine("using Libptx.Edsl.Common.Types.Scalar;");
                w_vec.WriteLineNoTabs(String.Empty);
                w_vec.WriteLine("namespace {0}", dir2ns(dir_vectors));
                w_vec.WriteLine("{");
                w_vec.Indent++;
                w_vec.WriteLine("public class {0} : vector", name);
                w_vec.WriteLine("{");
                w_vec.Indent++;

                Func<int, String> arg_name = i => i == 1 ? "x" : i == 2 ? "y" : i == 3 ? "z" : i == 4 ? "w" : 
                    ((Func<String>)(() => { throw AssertionHelper.Fail(); }))();
                var args = 1.UpTo(t.vec_rank()).Select(i => String.Format("reg_{0} {1}", t.vec_el().Name.Signature(), arg_name(i))).StringJoin(", ");
                w_vec.WriteLine("public {0}({1})", name, args);

                var arg_names = 1.UpTo(t.vec_rank()).Select(arg_name).StringJoin(", ");
                shortcuts[t.vec_rank()].Add(String.Format("public static {0} v{1}({2}) {{ return new {0}({3}); }}", name, t.vec_rank(), args, arg_names));

                w_vec.WriteLine("{");
                w_vec.Indent++;
                w_vec.WriteLine("ElementType = {0};", t.vec_el().Name.Signature());
                1.UpTo(t.vec_rank()).ForEach(i => w_vec.WriteLine("Elements.Add({0}.AssertCast<var>());", arg_name(i)));
                w_vec.Indent--;
                w_vec.WriteLine("}");

                w_vec.WriteLineNoTabs(String.Empty);
                w_vec.EmitTypeSpec(name, t, Space.Reg, true);
                w_vec.Indent--;

                w_vec.WriteLine("}");
                w_vec.Indent--;
                w_vec.WriteLine("}");

                var fname_reg = dir_vectors + name + ".cs";
                if (!Directory.Exists(dir_vectors)) Directory.CreateDirectory(dir_vectors);
                File.WriteAllText(fname_reg, buf_vec.ToString());
            });

            var buf = new StringBuilder();
            var w = new StringWriter(buf).Indented();
            w.WriteLine("using {0};", typeof(Vector).Namespace);
            w.WriteLine("using Libptx.Edsl.Common.Types.Scalar;");
            w.WriteLineNoTabs(String.Empty);
            w.WriteLine("namespace {0}", dir2ns(dir_vectors));
            w.WriteLine("{");
            w.Indent++;
            w.WriteLine("public class vector : Vector");
            w.WriteLine("{");
            w.Indent++;

            var write_shortcuts = shortcuts.Keys.Select<int, Action>(i => () => shortcuts[i].ForEach(w.WriteLine));
            var write_empty_lines = Seq.Nats.Select<int, Action>(_ => () => w.WriteLineNoTabs(String.Empty));
            write_shortcuts.Intersperse(write_empty_lines).SkipLast(1).RunEach();

            w.Indent--;
            w.WriteLine("}");
            w.Indent--;
            w.WriteLine("}");

            var fname = dir_vectors + "vector.cs";
            if (!Directory.Exists(dir_vectors)) Directory.CreateDirectory(dir_vectors);
            File.WriteAllText(fname, buf.ToString());
        }
    }
}