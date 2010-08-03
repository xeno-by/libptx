using System;
using System.IO;
using System.Linq;
using System.Text;
using Libptx.Expressions;
using Libptx.Reflection;
using XenoGears.Assertions;
using XenoGears.Strings;
using XenoGears.Functional;
using Libptx.Common.Types;
using XenoGears.Strings.Writers;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class TypeGenerator
    {
        public static void DoGenerate()
        {
            var edsl_base = @"..\..\..\..\Libptx.Edsl\Libptx.Edsl\";
            var dir_types = edsl_base + @"Common\Types\";
            Func<String, String> dir2ns = dir =>
            {
                var rel = dir.Replace(@"..\..\..\..\Libptx.Edsl\", String.Empty);
                rel = (rel + @"\").Unfold(s => s.Slice(0, -1), s => s.EndsWith(@"\")).Last();
                return rel.Replace(@"\", ".").Slice(0, -1);
            };

            Types.Opaque.ForEach(t =>
            {
                var dir_opaques = dir_types + @"Opaque\";

                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.WriteLine("using {0};", typeof(Expression).Namespace);
                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("namespace {0}", dir2ns(dir_opaques));
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0} : typed_expr", t);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public {0}(Expression expr)", t);
                w.Indent++;
                w.WriteLine(": base(expr)");
                w.Indent--;
                w.WriteLine("{");
                w.Indent++;
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");

                var fname = dir_opaques + t + ".cs";
                if (!Directory.Exists(dir_opaques)) Directory.CreateDirectory(dir_opaques);
                File.WriteAllText(fname, buf.ToString());
            });

            Types.Other.ForEach(t =>
            {
                var dir_opaques = dir_types + @"Other\";

                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.WriteLine("using {0};", typeof(Expression).Namespace);
                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("namespace {0}", dir2ns(dir_opaques));
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0} : typed_expr", t);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public {0}(Expression expr)", t);
                w.Indent++;
                w.WriteLine(": base(expr)");
                w.Indent--;
                w.WriteLine("{");
                w.Indent++;
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");

                var fname = dir_opaques + t + ".cs";
                if (!Directory.Exists(dir_opaques)) Directory.CreateDirectory(dir_opaques);
                File.WriteAllText(fname, buf.ToString());
            });

            var scalars = Combinatorics.CartesianProduct(Types.Scalar, new []{ null, "reg" }, new []{null, "relaxed"}).ToReadOnly();
            scalars.Zip((t, mod1, mod2) =>
            {
                if (mod1 == null && mod2 == "relaxed") return;
                var mod = mod1 == null && mod2 == null ? null :
                    mod1 == "reg" && mod2 == null ? "reg" :
                    mod1 == "reg" && mod2 == "relaxed" ? "relaxed_reg" :
                    ((Func<String>)(() => { throw AssertionHelper.Fail(); }))();
                var name = t.Name.Signature();
                if (mod != null) name = mod + "_" + name;
                var dir = dir_types + @"Scalar\";

                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.WriteLine("using {0};", typeof(Expression).Namespace);
                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("namespace {0}", dir2ns(dir));
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0} : typed_expr", name);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public {0}(Expression expr)", name);
                w.Indent++;
                w.WriteLine(": base(expr)");
                w.Indent--;
                w.WriteLine("{");
                w.Indent++;
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");

                var fname = dir + name + ".cs";
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(fname, buf.ToString());
            });

            var vectors = Combinatorics.CartesianProduct(Types.Vector, new []{ null, "reg" }, new []{null, "relaxed"}).ToReadOnly();
            vectors.Zip((t, mod1, mod2) =>
            {
                if (mod1 == null && mod2 == "relaxed") return;
                var mod = mod1 == null && mod2 == null ? null :
                    mod1 == "reg" && mod2 == null ? "reg" :
                    mod1 == "reg" && mod2 == "relaxed" ? "relaxed_reg" :
                    ((Func<String>)(() => { throw AssertionHelper.Fail(); }))();
                var name = String.Format("v{0}_{1}", t.vec_rank(), t.vec_el().Name.Signature());
                if (mod != null) name = mod + "_" + name;
                var dir = dir_types + @"Vector\";

                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.WriteLine("using {0};", typeof(Expression).Namespace);
                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("namespace {0}", dir2ns(dir));
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public partial class {0} : typed_expr", name);
                w.WriteLine("{");
                w.Indent++;
                w.WriteLine("public {0}(Expression expr)", name);
                w.Indent++;
                w.WriteLine(": base(expr)");
                w.Indent--;
                w.WriteLine("{");
                w.Indent++;
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");
                w.Indent--;
                w.WriteLine("}");

                var fname = dir + name + ".cs";
                if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
                File.WriteAllText(fname, buf.ToString());
            });
        }
    }
}