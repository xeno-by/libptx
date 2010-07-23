using System;
using System.IO;
using System.Text;
using Libcuda.Versions;
using Libptx.Edsl.TextGenerators.AdHoc;
using Libptx.Instructions;
using System.Linq;
using XenoGears.Functional;
using XenoGears.Reflection.Shortcuts;
using Libptx.Common.Annotations;
using XenoGears.Strings;

namespace Libptx.Edsl.TextGenerators
{
    internal class Program
    {
        public static void Main(String[] args)
        {
//            using (new Context(SoftwareIsa.PTX_21, HardwareIsa.SM_20))
//            {
//                PtxoptypeGenerator.DoGenerate();
//                SpecialtypeGenerator.DoGenerate();
//                TypeGenerator.DoGenerate();
//                SpecialGenerator.DoGenerate();
//                VectorGenerator.DoGenerate();
//            }

            var libptx = typeof(ptxop).Assembly;
            var ops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();
            foreach (var op in ops)
            {
//                var ps = op.GetProperties(BF.PublicInstance | BF.DeclOnly);
//                if (ps.Count(p => p.PropertyType == typeof(Type)) > 1)
//                {
//                    Console.WriteLine(op.Name);
//                }
//                var cnts = op.Signatures().Select(s => s.Count(c => c == ',')).Distinct();
//                if (cnts.Count() > 1)
//                {
//                    Console.WriteLine(op.Name + ": " + cnts.StringJoin());
//                    if (op.Name == "mad")
//                    {
//                        op.Signatures().ForEach(s => Console.WriteLine(s));
//                    }
//                }

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
                    var parsed_args = arg_list.Split(',').Select(arg =>
                    {
                        arg = arg.Trim().Extract(@"^(\})?(?<name>.*)(\{)?$");
                        var parsed = arg.Trim().Parse(@"^((\{)?(?<prefix>[!-])(\})?)?(?<name>[\w\d]+)(\[\|(?<othername>[\w\d]+)\])?((\{)?\.(?<suffix>[\w\d]+)(\})?)?$");
                        return parsed["name"];
                    }).ToReadOnly();

                    return parsed_args;
                }).Distinct().ToReadOnly();
                names.ForEach(name => w.WriteLine("public Type {0} {{ get; set; }}", name));

                w.WriteLineNoTabs(String.Empty);
                w.WriteLine("protected override void custom_validate_ops(SoftwareIsa target_swisa, HardwareIsa target_hwisa)");
                w.WriteLine("{");
                w.Indent++;
                names.ForEach(name =>
                {
                    var props = op.GetProperties(BF.PublicInstance).Where(p => p.PropertyType == typeof(Type)).Select(p => p.Name).ToHashSet();

                    var prop = "N/A" as String;
                    if (props.Contains(name + "type")) prop = name + "type";
                    if (props.Contains("type")) prop = "type";

                    w.WriteLine("({0} == {1}).AssertTrue();", name, prop);
                });
                w.Indent--;
                w.WriteLine("}");

                var dir = @"..\..\..\..\Libptx\" + op.Namespace.Replace(".", @"\") + @"\";
                var file = dir + op.Name + ".cs";
                var text = File.ReadAllText(file);
                var iof = text.IndicesOf(c => c == '}').ThirdLast();
                text = text.Insert(iof, buf.ToString());
                File.WriteAllText(file, text);
            }
        }
    }
}