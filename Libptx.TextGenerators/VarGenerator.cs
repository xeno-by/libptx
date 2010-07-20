using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Libcuda.DataTypes;
using Libptx.Common.Types;
using XenoGears;
using XenoGears.Assertions;
using XenoGears.Functional;
using XenoGears.Strings;
using Type=System.Type;

namespace Libptx.TextGenerators
{
    internal static class VarGenerator
    {
        public static void DoGenerate()
        {
            var types_file = @"..\..\..\Libptx\Common\Types\TypeName.cs";
            var types_lines = File.ReadAllLines(types_file);
            var types = types_lines.Select(s => s.Extract(@"^\s*\[Affix\("".+?""(,\s*.+?\s*)*\)\]\s+(?<name>[\d\w]+?)(\s*=\s*.+?)?\s*,\s*$")).Where(p => p != null).ToReadOnly();
            types = types.Select(s => s.ToLower()).ToReadOnly();

            var clr_types = new Dictionary<String, Type>();
            clr_types.Add("u8", typeof(byte));
            clr_types.Add("s8", typeof(sbyte));
            clr_types.Add("u16", typeof(ushort));
            clr_types.Add("s16", typeof(short));
            clr_types.Add("u32", typeof(uint));
            clr_types.Add("s32", typeof(int));
            clr_types.Add("u64", typeof(ulong));
            clr_types.Add("s64", typeof(long));
            clr_types.Add("f16", typeof(half));
            clr_types.Add("f32", typeof(float));
            clr_types.Add("f64", typeof(double));
            clr_types.Add("b8", typeof(Bit8));
            clr_types.Add("b16", typeof(Bit16));
            clr_types.Add("b32", typeof(Bit32));
            clr_types.Add("b64", typeof(Bit64));
            clr_types.Add("pred", typeof(bool));
            clr_types.Add("tex", typeof(Tex));
            clr_types.Add("sampler", typeof(Sampler));
            clr_types.Add("surf", typeof(Surf));
            Func<String, Type> type2clr = s =>
            {
                var name = s.Split('_').First();
                var mods = s.Split('_').Skip(1).ToReadOnly();
                var baseline = clr_types[name];
                var result = mods.Fold(baseline, (t, mod) =>
                {
                    var sv = mod.Extract(@"^v(?<len>\d*)$");
                    if (sv != null)
                    {
                        var iv = int.Parse(sv);
                        var tv1_name = typeof(int3).Namespace + "." + t.GetCSharpRef(ToCSharpOptions.Terse) + iv;
                        var tv1 = typeof(int3).Assembly.GetType(tv1_name);
                        var tv2_name = typeof(Bit8).Namespace + "." + t.GetCSharpRef(ToCSharpOptions.Terse) + iv;
                        var tv2 = typeof(Bit8).Assembly.GetType(tv2_name);
                        var tv3_name = typeof(Bit8).Namespace + "." + t.GetCSharpRef(ToCSharpOptions.Terse) + "_V" + iv;
                        var tv3 = typeof(Bit8).Assembly.GetType(tv3_name);
                        return tv1 ?? tv2 ?? tv3;
                    }

                    var sa = mod.Extract(@"^a(?<len>\d*)$");
                    if (sa != null)
                    {
                        var ia = int.Parse(sa);
                        var ta = 1.UpTo(ia).Fold(t, (aux, _) => aux == null ? null : aux.MakeArrayType());
                        return ta;
                    }

                    throw AssertionHelper.Fail();
                });

                return result;
            };

            var spaces_file = @"..\..\..\Libptx\Common\Enumerations\Space.cs";
            var spaces_lines = File.ReadAllLines(spaces_file);
            var spaces = spaces_lines.Select(s => s.Parse(@"^\s*\[Affix\(""(?<sig>.+?)""(,\s*.+?\s*)*\)\]\s+(?<name>[\d\w]+?)(\s*=\s*.+?)?\s*,\s*$")).Where(p => p != null)
                .ToDictionary(p => p["sig"].Replace("[", "").Replace("]", "").Fluent(s => s == "const" ? "@const" : s), p => p["name"]).ToReadOnly();

            var dir_types = @"..\..\..\Libptx\Edsl\Types\";
            if (!Directory.Exists(dir_types)) Directory.CreateDirectory(dir_types);
            var dir_types_relaxed = @"..\..\..\Libptx\Edsl\Types\Relaxed\";
            if (!Directory.Exists(dir_types_relaxed)) Directory.CreateDirectory(dir_types_relaxed);
            var dir_vars = @"..\..\..\Libptx\Edsl\Vars\";
            if (!Directory.Exists(dir_vars)) Directory.CreateDirectory(dir_vars);
            var dir_hastypes = @"..\..\..\Libptx\Edsl\Vars\Types\";
            if (!Directory.Exists(dir_hastypes)) Directory.CreateDirectory(dir_hastypes);
            Func<String, String> dir2ns = dir => dir.Replace(@"..\..\..\", String.Empty).Replace(@"\", ".").Slice(0, -1);

            Func<Type, bool> is_vec = clr =>
            {
                if (clr.IsCudaVector()) return true;
                if (clr == typeof(ulong3) || clr == typeof(ulong4)) return true;
                if (clr == typeof(long3) || clr == typeof(long4)) return true;
                if (clr == typeof(double3) || clr == typeof(double4)) return true;

                var m1 = clr.FullName.Extract(@"^(?<name>.*)[1-4]$");
                if (m1 != null)
                {
                    var t1 = clr.Assembly.GetType(m1);
                    if (t1 != null) return true;
                }

                var m2 = clr.FullName.Extract(@"^(?<name>.*)_[v|V][1-4]$");
                if (m2 != null)
                {
                    var t2 = clr.Assembly.GetType(m2);
                    if (t2 != null) return true;
                }

                return false;
            };

            #region var

//            const int max_bits = max_bits;
            const int max_bits = int.MaxValue;

            Func<String, String> var_fa = type =>
            {
                var plain = type == "pred" || type == "tex" || type == "sampler" || type == "surf";

                var buf_var_fa = new StringBuilder();
                var w_var_fa = new StringWriter(buf_var_fa).Indented();
                w_var_fa.Indent += 2;
                if (!plain)
                {
                    Action<String> write_vec = vec => w_var_fa.WriteLine("public var_{0}_{1} {1} {{ get {{ return Clone<var_{0}_{1}>(v => v.Type = v.Type.{1}, v => v.Init = null); }} }}", type, vec);
                    Action write_arr1 = () => w_var_fa.WriteLine("public var_{0}_a1 this[int dim] {{ get {{ return Clone<var_{0}_a1>(v => v.Type = v.Type[dim], v => v.Init = null); }} }}", type);
//                    Action write_arr11 = () => w_var_fa.WriteLine("public var_{0}_a2 this[int dim] {{ get {{ return Clone<var_{0}_a2>(v => v.Type = v.Type[dim], v => v.Init = null); }} }}", type.Replace("_a1", ""));
//                    Action write_arr2 = () => w_var_fa.WriteLine("public var_{0}_a2 this[int dim1, int dim2] {{ get {{ return Clone<var_{0}_a2>(v => v.Type = v.Type[dim1, dim2], v => v.Init = null); }} }}", type);
                    Action<String> write_vecf = fld => w_var_fa.WriteLine("public var_{0} {1} {{ get {{ return Clone<var_{0}>(v => v.Type = v.Type.{1}, v => v.Init = null); }} }}", type.Replace("_v1", "").Replace("_v2", "").Replace("_v4", ""), fld);

                    var clr = type2clr(type);
                    var is_arr = clr.IsArray;
                    var is_arr1 = is_arr && !clr.GetElementType().IsArray;
//                    var is_arr2 = is_arr && !is_arr1 && !clr.GetElementType().GetElementType().IsArray;
                    var is_scalar = !is_vec(clr) && !is_arr;

                    var el = clr.Unfold(t => t.GetElementType(), t => t != null).Last();
                    var sz = Marshal.SizeOf(el) * 8;
                    sz *= (is_vec(clr) ? int.Parse(clr.Name.Slice(-1)) : 1);

                    if (is_scalar && sz * 1 <= max_bits) write_vec("v1");
                    if (is_scalar && sz * 2 <= max_bits) write_vec("v2");
                    if (is_scalar && sz * 4 <= max_bits) write_vec("v4");
                    if (!is_arr) write_arr1();
//                    if (!is_arr) write_arr2();
//                    if (is_arr1) write_arr11();

                    var vec_sz = is_vec(clr) ? int.Parse(clr.Name.Slice(-1)) : 0;
                    if (is_vec(clr) && vec_sz >= 1) write_vecf("x");
                    if (is_vec(clr) && vec_sz >= 1) write_vecf("r");
                    if (is_vec(clr) && vec_sz >= 2) write_vecf("y");
                    if (is_vec(clr) && vec_sz >= 2) write_vecf("g");
                    if (is_vec(clr) && vec_sz >= 3) write_vecf("z");
                    if (is_vec(clr) && vec_sz >= 3) write_vecf("b");
                    if (is_vec(clr) && vec_sz >= 4) write_vecf("w");
                    if (is_vec(clr) && vec_sz >= 4) write_vecf("a");
                }

                return buf_var_fa.ToString().TrimEnd();
            };

            Func<String, String> var_common = type =>
            {
                var buf_var_common = new StringBuilder();
                var w_var_common = new StringWriter(buf_var_common).Indented();
                w_var_common.Indent += 2;

                // var_u16 (spaces)
                spaces.ForEach(space => w_var_common.WriteLine("public new var_{0} {1} {{ get {{ return Clone(v => v.Space = Common.Enumerations.space.{2}); }} }}", type, space.Key, space.Value));
                w_var_common.WriteLineNoTabs(String.Empty);

                // var_u16 (init)
                var clr = type2clr(type);
                w_var_common.WriteLine("public var_{0} init({1} value) {{ return Clone(v => v.Init = value); }}", type, clr.GetCSharpRef(ToCSharpOptions.Terse));
                if (is_vec(clr) && clr.Name.EndsWith("4"))
                {
                    var vec3_name = clr.FullName.Slice(0, -1) + "3";
                    var vec3 = clr.Assembly.GetType(vec3_name).AssertNotNull();
                    w_var_common.WriteLine("public var_{0} init({1} value) {{ return Clone(v => v.Init = value); }}", type, vec3.GetCSharpRef(ToCSharpOptions.Terse));
                }
                w_var_common.WriteLineNoTabs(String.Empty);

                // var_u16 (alignment)
                var el_type = clr.Unfold(t => t.GetElementType(), t => t != null).Last();
                w_var_common.WriteLine("public var_{0}() {{ Alignment = {1} /* sizeof({2}) */; }}", type, Marshal.SizeOf(el_type), el_type.GetCSharpRef(ToCSharpOptions.Terse));
                w_var_common.WriteLine("public var_{0} align(int alignment){{ return Clone(v => v.Alignment = alignment.AssertThat(a => a.Unfoldi(i => i / 2, i => i > 1).All(mod => mod == 0))); }}", type);
                var default_align = Marshal.SizeOf(el_type);
                new[] { 1, 2, 4, 8 }.Select(i => i * default_align).ForEach(align => w_var_common.WriteLine("public var_{0} align{1}{{ get {{ return align({1}); }} }}", type, align));
                w_var_common.WriteLineNoTabs(String.Empty);

                // var_u16 export
                w_var_common.WriteLine("public var_{0} export {{ get {{ return Clone(v => v.IsVisible = true); }} }}", type);
                w_var_common.WriteLine("public var_{0} import {{ get {{ return Clone(v => v.IsExtern = true); }} }}", type);
                w_var_common.WriteLine("public var_{0} @extern {{ get {{ return Clone(v => v.IsExtern = true); }} }}", type);

                return buf_var_common.ToString();
            };

            Func<String, String> var_clones = type =>
            {
                var buf_var_clones = new StringBuilder();
                var w_var_clones = new StringWriter(buf_var_clones).Indented();
                var var_clones_template = @"
        internal Var_U16 Clone()
        {
            return Clone<Var_U16>();
        }

        internal T Clone<T>()
            where T : var, new()
        {
            T clone = new T();
            clone.Name = this.Name;
            clone.Space = this.Space;
            clone.Type = this.Type;
            clone.Init = this.Init;
            clone.Alignment = this.Alignment;
            clone.Mod = this.Mod;
            clone.IsVisible = this.IsVisible;
            clone.IsExtern = this.IsExtern;
            return clone;
        }

        internal Var_U16 Clone(params Action<Var_U16>[] mods)
        {
            return Clone<Var_U16>(mods);
        }

        internal T Clone<T>(params Action<T>[] mods)
            where T : var, new()
        {
            T clone = Clone<T>();
            foreach (Action<T> mod in mods) mod(clone);
            return clone;
        }";
                var_clones_template = var_clones_template.Replace("Var_U16", "var_" + type).Slice(Environment.NewLine.Length);
                w_var_clones.Write(var_clones_template);
                return buf_var_clones.ToString();
            };

            Func<String, String> var = type =>
            {
                var buf_var = new StringBuilder();
                var w_var = new StringWriter(buf_var).Indented();
                w_var.WriteLine("using System;");
                w_var.WriteLine("using System.Linq;");
                w_var.WriteLine("using Libptx.Common.Types;");
                w_var.WriteLine("using Libptx.Edsl.Vars.Types;");
                w_var.WriteLine("using Libptx.Expressions;");
                w_var.WriteLine("using Libcuda.DataTypes;");
                w_var.WriteLine("using XenoGears.Assertions;");
                w_var.WriteLine("using XenoGears.Functional;");
                w_var.WriteLine("");
                w_var.WriteLine("namespace {0}", dir2ns(dir_vars));
                w_var.WriteLine("{");
                w_var.Indent++;
                w_var.WriteLine("public class var_{0} : var", type.ToLower());
                w_var.WriteLine("{");
                w_var.Indent++;

                if (type == "u32")
                {
                    w_var.WriteLine("public static var_u32 operator -(var_u32 var_u32) { return var_u32.Clone(v => v.Mod |= VarMod.Neg); }");
                    w_var.WriteLine("public var_u32 b0 { get { return Clone(v => v.Mod |= VarMod.B0); } }");
                    w_var.WriteLine("public var_u32 b1 { get { return Clone(v => v.Mod |= VarMod.B1); } }");
                    w_var.WriteLine("public var_u32 b2 { get { return Clone(v => v.Mod |= VarMod.B2); } }");
                    w_var.WriteLine("public var_u32 b3 { get { return Clone(v => v.Mod |= VarMod.B3); } }");
                    w_var.WriteLine("public var_u32 h0 { get { return Clone(v => v.Mod |= VarMod.H0); } }");
                    w_var.WriteLine("public var_u32 h1 { get { return Clone(v => v.Mod |= VarMod.H1); } }");
                    w_var.WriteLineNoTabs(String.Empty);
                }
                else if (type == "s32")
                {
                    w_var.WriteLine("public static var_s32 operator -(var_s32 var_s32) { return var_s32.Clone(v => v.Mod |= VarMod.Neg); }");
                    w_var.WriteLine("public var_s32 b0 { get { return Clone(v => v.Mod |= VarMod.B0); } }");
                    w_var.WriteLine("public var_s32 b1 { get { return Clone(v => v.Mod |= VarMod.B1); } }");
                    w_var.WriteLine("public var_s32 b2 { get { return Clone(v => v.Mod |= VarMod.B2); } }");
                    w_var.WriteLine("public var_s32 b3 { get { return Clone(v => v.Mod |= VarMod.B3); } }");
                    w_var.WriteLine("public var_s32 h0 { get { return Clone(v => v.Mod |= VarMod.H0); } }");
                    w_var.WriteLine("public var_s32 h1 { get { return Clone(v => v.Mod |= VarMod.H1); } }");
                    w_var.WriteLineNoTabs(String.Empty);
                }
                else if (type == "pred")
                {
                    w_var.WriteLine("public static var_pred operator !(var_pred var_pred) { return var_pred.Clone(v => v.Mod |= VarMod.Not); }");
                    w_var.WriteLine("public static var_couple operator |(var_pred var_pred1, var_pred var_pred2) { return new var_couple{fst = var_pred1, snd = var_pred2}; }");
                    w_var.WriteLineNoTabs(String.Empty);
                }

                var fa = var_fa(type);
                if (fa.IsNotEmpty()) w_var.WriteLineNoTabs(fa);
                w_var.WriteLineNoTabs(var_common(type));
                w_var.WriteLineNoTabs(var_clones(type));

                w_var.Indent--;
                w_var.WriteLine("}");
                w_var.Indent--;
                w_var.WriteLine("}");
                return buf_var.ToString();
            };

            #endregion

            foreach (var type in types)
            {
                /////////////////// var_u16 ///////////////////////

                var plain = type == "pred" || type == "tex" || type == "sampler" || type == "surf";
                var vecs = new []{null, "v1", "v2", "v4"};
//                var arrs = new []{null, "a1", "a2"};
                var arrs = new []{null, "a1"};
                var mods = Combinatorics.CartesianProduct(vecs, arrs);
                var c_types = plain ? type.MkArray() : mods.Zip((mod1, mod2) =>
                {
                    var buf = new StringBuilder(type);
                    if (mod1 != null) buf.Append("_" + mod1);
                    if (mod2 != null) buf.Append("_" + mod2);
                    return buf.ToString();
                }).ToArray();

                c_types.ForEach(c_type =>
                {
                    var is_scalar = !c_type.Contains("_");
                    var clr = type2clr(c_type);
                    if (clr == null) return;

                    /////////////////// var_u16 ///////////////////////

                    var var_fname = dir_vars + "var_" + c_type + ".cs";
                    var var_text = var(c_type);
                    if (is_scalar) var_text = var_text.Replace("var_" + c_type + " : var", "var_" + c_type + " : has_type_" + c_type);
                    File.WriteAllText(var_fname, var_text);

                    if (is_scalar)
                    {
                        /////////////////// u16 ///////////////////////

                        var buf_type = new StringBuilder();
                        var w_type = new StringWriter(buf_type).Indented();
                        w_type.WriteLine("using System.Linq;");
                        w_type.WriteLine("using Libptx.Common.Types;");
                        w_type.WriteLine("using Libptx.Edsl.Vars;");
                        w_type.WriteLine("using Libcuda.DataTypes;");
                        w_type.WriteLine("using XenoGears.Assertions;");
                        w_type.WriteLine("using XenoGears.Functional;");
                        w_type.WriteLine("");
                        w_type.WriteLine("namespace {0}", dir2ns(dir_types));
                        w_type.WriteLine("{");
                        w_type.Indent++;
                        w_type.WriteLine("public class {0} : type", c_type);
                        w_type.WriteLine("{");
                        w_type.Indent++;

                        var buf = new StringBuilder();
                        var w = new StringWriter(buf);
                        var fa = var_fa(c_type);
                        fa = fa.Replace("public", "public static new");
                        if (fa.IsNotEmpty()) w.WriteLine(fa);
                        var cmn = var_common(c_type);
                        cmn = cmn.Replace("public new", "public").Replace("public", "public static");
                        w.WriteLine(cmn);
                        var s = buf.ToString();
                        s = s.Replace("return Clone", "return new var_" + c_type + "().Clone");
                        s = s.SplitLines().Select(ln =>
                        {
                            if (ln.Contains("this")) return "        // " + ln.Trim();
                            if (ln.Contains("{ Alignment =")) return null;
                            return ln;
                        }).Where(ln => ln != null).StringJoin(Environment.NewLine);
                        w_type.WriteLineNoTabs(s.TrimEnd());

                        w_type.Indent--;
                        w_type.WriteLine("}");
                        w_type.Indent--;
                        w_type.WriteLine("}");
                        var type_fname = dir_types + c_type + ".cs";
                        File.WriteAllText(type_fname, buf_type.ToString());

                        /////////////////// relaxed_u16 ///////////////////////

                        if ((c_type.StartsWith("b") || c_type.StartsWith("u") || c_type.StartsWith("s") || c_type.StartsWith("f")) &&
                            (c_type != "sampler" && c_type != "surf"))
                        {
                            var buf_relaxed = new StringBuilder();
                            var w_relaxed = new StringWriter(buf_relaxed).Indented();
                            w_relaxed.WriteLine("namespace {0}", dir2ns(dir_types_relaxed));
                            w_relaxed.WriteLine("{");
                            w_relaxed.Indent++;
                            w_relaxed.WriteLine("public class relaxed_{0} : relaxed_type<{1}>", c_type, c_type);
                            w_relaxed.WriteLine("{");
                            w_relaxed.Indent++;
                            w_relaxed.Indent--;
                            w_relaxed.WriteLine("}");
                            w_relaxed.Indent--;
                            w_relaxed.WriteLine("}");

                            var relaxed_fname = dir_types_relaxed + "relaxed_" + c_type + ".cs";
                            File.WriteAllText(relaxed_fname, buf_relaxed.ToString());
                        }

                        /////////////////// has_type_u16 ///////////////////////

                        var buf_has_type = new StringBuilder();
                        var w_has_type = new StringWriter(buf_has_type).Indented();
                        w_has_type.WriteLine("using Libptx.Edsl.Types;");
                        w_has_type.WriteLine("using Libptx.Edsl.Types.Relaxed;");
                        w_has_type.WriteLine();
                        w_has_type.WriteLine("namespace {0}", dir2ns(dir_hastypes));
                        w_has_type.WriteLine("{");
                        w_has_type.Indent++;
                        w_has_type.WriteLine("public class has_type_{0} : has_type<{0}>", c_type);
                        w_has_type.WriteLine("{");
                        w_has_type.Indent++;

                        var i = (c_type.StartsWith("u") || c_type.StartsWith("s")) && c_type == "surf" && c_type == "sampler";
                        var f = c_type.StartsWith("f");
                        var b = c_type.StartsWith("b");
                        if (i || f || b)
                        {
                            var relaxed = new List<String>();
                            var strict = new List<String>();

                            var sz = int.Parse(c_type.Slice(1));
                            var s_sizes = sz.MkArray();
                            var r_sizes = (new[] { 8, 16, 32, 64 }).Where(sz1 => sz1 <= sz).Reverse().ToReadOnly();

                            if (i)
                            {
                                s_sizes.ForEach(sz1 => strict.Add("u" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("u" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("s" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("s" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("b" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("b" + sz1));
                            }
                            else if (f)
                            {
                                s_sizes.ForEach(sz1 => strict.Add("f" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("f" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("b" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("b" + sz1));
                            }
                            else
                            {
                                b.AssertTrue();

                                s_sizes.ForEach(sz1 => strict.Add("u" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("u" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("s" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("s" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("f" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("f" + sz1));
                                s_sizes.ForEach(sz1 => strict.Add("b" + sz1));
                                r_sizes.ForEach(sz1 => relaxed.Add("b" + sz1));
                            }

                            strict.Remove("f8");
                            relaxed.Remove("f8");

                            strict.ForEach(t => w_has_type.WriteLine("public static implicit operator {0}(has_type_{1} var_{1}) {{ return new {0}{{var = var_{1}}}; }}", t, c_type));
                            relaxed.ForEach(t => w_has_type.WriteLine("public static implicit operator relaxed_{0}(has_type_{1} var_{1}) {{ return new relaxed_{0}{{var = var_{1}}}; }}", t, c_type));
                        }

                        w_has_type.Indent--;
                        w_has_type.WriteLine("}");
                        w_has_type.Indent--;
                        w_has_type.WriteLine("}");
                        var has_type_fname = dir_hastypes + "has_type_" + c_type + ".cs";
                        File.WriteAllText(has_type_fname, buf_has_type.ToString());
                    }
                });
            }
        }
    }
}