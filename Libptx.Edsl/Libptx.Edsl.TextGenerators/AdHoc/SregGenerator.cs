using System;
using System.IO;
using System.Text;
using Libcuda.Versions;
using Libptx.Common.Types;
using Libptx.Expressions.Sregs;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Functional;
using XenoGears.Strings;
using System.Linq;
using Libptx.Edsl.TextGenerators.Common;
using XenoGears.Reflection.Attributes;
using XenoGears.Strings.Writers;
using Type=Libptx.Common.Types.Type;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class SregGenerator
    {
        public static void DoGenerate()
        {
            var edsl_base = @"..\..\..\..\Libptx.Edsl\Libptx.Edsl\";
            var dir_specials = edsl_base + @"Expressions\Sregs\";
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\..\Libptx.Edsl\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            var libptx = typeof(Sreg).Assembly;
            var specials = libptx.GetTypes().Where(t => t.BaseType == typeof(Sreg)).ToReadOnly();
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

                var xid = t.Name == "tid" || t.Name == "ntid" || t.Name == "ctaid" || t.Name == "nctaid";
                if (xid)
                {
                    var post_20 = Context.Current.Version >= SoftwareIsa.PTX_20;
                    if (post_20)
                    {
                        w.EmitTypeSpec(t.Name, new Type { Name = TypeName.U16, Mod = TypeMod.V4 }, Space.Other);
                        w.EmitTypeSpec(t.Name, new Type { Name = TypeName.U32, Mod = TypeMod.V4 }, Space.Other);
                    }
                    else
                    {
                        w.EmitTypeSpec(t.Name, new Type { Name = TypeName.U16, Mod = TypeMod.V4 }, Space.Other);
                    }
                }
                else
                {
                    var type = t.Attr<SregAttribute>().Type;
                    w.EmitTypeSpec(t.Name, type, Space.Other);
                }

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