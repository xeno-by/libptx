using System;
using System.IO;
using System.Text;
using Libptx.Instructions;
using Libptx.Reflection;
using System.Linq;
using XenoGears.Functional;
using XenoGears.Reflection.Shortcuts;
using XenoGears.Strings;
using XenoGears.Strings.Writers;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Edsl.TextGenerators.AdHoc
{
    internal static class PtxopGenerator
    {
        public static void DoGenerate()
        {
            var libptx = typeof(ptxop).Assembly;
            var ops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();
            foreach (var op in ops)
            {
                var dir = @"..\..\..\..\" + op.Namespace.Replace(".", @"\") + @"\";
                var file = dir + op.Name + ".cs";
                var text = File.ReadAllText(file);
                Action<String> use = ns =>
                {
                    if (!text.Contains("using " + ns + ";"))
                    {
                        var liof = text.LastIndexOf("using") + 1;
                        var next = text.IndexOf(Environment.NewLine, liof);
                        var ins = next == -1 ? 0 : (next + Environment.NewLine.Length);
                        text = text.Insert(ins, "using " + ns + ";" + Environment.NewLine);
                    }
                };

                var buf = new StringBuilder();
                var w = new StringWriter(buf).Indented();
                w.Indent += 2;
                w.WriteLineNoTabs(String.Empty);

                w.WriteLineNoTabs(String.Empty);
                var names = op.Signatures().SelectMany(s =>
                {
                    var bal = 0;
                    var eof = s.TakeWhile(c =>
                    {
                        if (c == '{') bal++;
                        if (c == '}') bal--;
                        if (bal == 0 && c == ' ') return false;
                        return true;
                    }).Count();

                    var arg_list = s.Slice(eof + 1).Trim();
                    arg_list = arg_list.Extract("^(?<name>.*?);?$");
                    var parsed_args = arg_list.Split(",".MkArray(), StringSplitOptions.RemoveEmptyEntries).Select(arg =>
                    {
                        arg = arg.Trim().Replace("[", "").Replace("]", "").Replace("{", "").Replace("}", "");
                        var parsed = arg.Trim().Parse(@"^(?<prefix>[!-])?(?<name>[\w\d]+)(\|(?<othername>[\w\d]+))?(\.(?<suffix>[\w\d]+))?$");
                        return parsed["name"];
                    }).ToReadOnly();

                    return parsed_args;
                }).Distinct().ToReadOnly();
                if (names.IsEmpty()) continue;

                names.ForEach(name =>
                {
                    use("Libptx.Expressions");
                    var decl = String.Format("public Expression {0} {{ get; set; }}", name);
                    if (text.Contains(decl))
                    {
                        var iof_decl = text.IndexOf(decl);
                        iof_decl -= "        ".Length;
                        iof_decl -= Environment.NewLine.Length * 2;
                        var iof_brace = text.IndexOf("}", iof_decl);
                        text = text.Remove(iof_decl, iof_brace - iof_decl + 1);
                    }

                    w.WriteLine(decl);
                });

                w.WriteLineNoTabs(String.Empty);
                use("Libcuda.Versions");
                w.WriteLine("protected override void custom_validate_ops(SoftwareIsa target_swisa, HardwareIsa target_hwisa)");
                w.WriteLine("{");
                w.Indent++;
                names.ForEach(name =>
                {
                    var props = op.GetProperties(BF.PublicInstance).Where(p => p.PropertyType == typeof(Type)).Select(p => p.Name).ToHashSet();

                    var prop = "N/A" as String;
                    if (props.Contains(name + "type")) prop = name + "type";
                    if (props.Contains("type")) prop = "type";

                    use("XenoGears.Assertions");
                    w.WriteLine("agree({0}, {1}).AssertTrue();", name, prop);
                });
                w.Indent--;
                w.Write("}");

                var iof = text.IndicesOf(c => c == '}').ThirdLastOrDefault(-1);
                if (iof == -1) continue;
                text = text.Insert(iof + 1, buf.ToString());
                File.WriteAllText(file, text);
            }
        }
    }
}