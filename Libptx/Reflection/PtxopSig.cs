using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using XenoGears.Functional;
using XenoGears.Strings;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;
using Type=System.Type;
using XenoGears;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopSig
    {
        public Type Decl { get; private set; }
        public PtxopAttribute Meta { get; private set; }

        public SoftwareIsa Version { get { return Meta.Version; } }
        public HardwareIsa Target { get { return Meta.Target; } }

        public String Opcode { get; private set; }
        public ReadOnlyCollection<PtxopMod> Mods { get; private set; }
        public ReadOnlyCollection<PtxopAffix> Affixes { get; private set; }
        public ReadOnlyCollection<PtxopOperand> Operands { get; private set; }

        internal PtxopSig(Type decl, PtxopAttribute meta)
        {
            Decl = decl;
            Meta = meta;

            var props = decl.GetProperties(BF.PublicInstance);
            var p_mods = props.Where(p => p.HasAttr<ModAttribute>()).ToDictionary(p => p.Signature(), p => p);
            var p_affixes = props.Where(p => p.HasAttr<AffixAttribute>()).ToDictionary(p => p.Signature(), p => p);
            var p_operands = props.Where(p => typeof(Expression).IsAssignableFrom(p.PropertyType)).ToDictionary(p => p.Name, p => p);

            var mods = new List<PtxopMod>();
            var affixes = new List<PtxopAffix>();
            var operands = new List<PtxopOperand>();
            int s_reading_opcode = 0, s_expecting_affix = 1, s_reading_affix = 2, s_expecting_operand = 3, s_reading_operand = 4, s_finish = 5;
            var state = s_reading_opcode;
            var quote = '\0';
            var buf = new StringBuilder();
            Action flush = () =>
            {
                var s = buf.ToString();
                buf = new StringBuilder();
                var mand = quote == '\0' || quote == '[';

                if (state == s_reading_opcode)
                {
                    s.AssertMatch(@"^\w+$");

                    var s_mod = p_mods.SingleOrDefault(kvp => s.EndsWith(kvp.Key)).Key;
                    if (s_mod != null)
                    {
                        s = s.Slice(0, -1 * s_mod.Length);

                        var p_mod = p_mods[s_mod];
                        p_mods.RemoveElements(kvp => kvp.Value == p_mod);

                        var meta_mod = p_mod.Attr<ModAttribute>();
                        mods.Add(new PtxopMod(p_mod, meta_mod, s_mod, true, null));
                    }

                    Opcode = s;
                }
                else if (state == s_reading_affix)
                {
                    var s_hardcoded = decl.Name.Fluent(s1 => { var iof = s1.IndexOf("_"); return iof == -1 ? String.Empty : s1.Slice(iof + 1).Replace("_", "."); });
                    var hardcoded = s_hardcoded.Split('.').Where(s1 => s1.IsNotEmpty()).ToHashSet();
                    if (hardcoded.Contains(s)) return;

                    if (p_affixes.ContainsKey(s))
                    {
                        s.AssertMatch(@"^\w+$");

                        var p_affix = p_affixes[s];
                        p_affixes.RemoveElements(kvp => kvp.Value == p_affix);

                        var meta_affix = p_affix.Attr<AffixAttribute>();
                        affixes.Add(new PtxopAffix(p_affix, meta_affix, s, mand));
                    }
                    else
                    {
                        PropertyInfo p_affix = null;
                        var options = s.Split(',').Select(s_opt => s_opt.Trim()).Select(s_opt =>
                        {
                            Func<PropertyInfo, String, Object> parse = (p, s1) =>
                            {
                                var t = p.PropertyType;
                                if (t.IsEnum)
                                {
                                    var values = Enum.GetValues(t).Cast<Object>();
                                    var value = values.SingleOrDefault(v => v.Signature() == s1);
                                    return value;
                                }
                                else if (t == typeof(Libptx.Common.Types.Type))
                                {
                                    var values = Enum.GetValues(typeof(TypeName)).Cast<TypeName>();
                                    var value = values.SingleOrDefault(v => v.Signature() == s1);
                                    return (Libptx.Common.Types.Type)value;
                                }
                                else
                                {
                                    return null;
                                }
                            };

                            s_opt.AssertMatch(@"^\w+$");
                            Object opt = null;
                            if (p_affix == null)
                            {
                                p_affixes.Values.ForEach(p =>
                                {
                                    var o = parse(p, s_opt);
                                    if (o != null)
                                    {
                                        (p_affix == null).AssertTrue();
                                        p_affix = p;
                                        opt = o;
                                    }
                                });
                            }
                            else
                            {
                                opt = parse(p_affix, s_opt);
                            }

                            (opt != null).AssertTrue();
                            return opt;
                        }).ToReadOnly();
                        if (options.IsEmpty()) options = null;

                        (p_affix != null).AssertTrue();
                        p_affixes.RemoveElements(kvp => kvp.Value == p_affix);

                        var meta_affix = p_affix.Attr<AffixAttribute>();
                        affixes.Add(new PtxopAffix(p_affix, meta_affix, p_affix.Signature(), mand, options));
                    }
                }
                else if (state == s_reading_operand)
                {
                    var ms = p_operands.Keys.ToDictionary(nm => nm, nm => s.Parse(@"^(?<prefix>.*?\W)?(?<name>" + nm + @")(?<suffix>\W.*)?$"));
                    var mo = ms.AssertSingle(kvp => kvp.Value != null).Value;

                    var name = mo["name"];
                    name.AssertMatch(@"^\w+$");

                    var optional_mods = new List<Mod>();
                    var mandatory_mods = new List<Mod>();
                    var prefix = mo["prefix"];
                    if (prefix.IsNotEmpty())
                    {
                        prefix = prefix.Extract(@"^\[(?<purified>.*)\]$") ?? prefix;
                        var purified = prefix.Extract(@"^\{(?<purified>.*)\}$");
                        prefix = purified ?? prefix;
                        Action<Mod> add = mod =>
                        {
                            if (purified == null) mandatory_mods.Add(mod);
                            else optional_mods.Add(mod);
                        };

                        if (prefix == "!")
                        {
                            add(Mod.Not);
                        }
                        else if (prefix == "-")
                        {
                            add(Mod.Neg);
                        }
                        else
                        {
                            throw AssertionHelper.Fail();
                        }
                    }

                    var suffix = mo["suffix"];
                    if (suffix.IsNotEmpty())
                    {
                        suffix = suffix.Extract(@"^\[(?<purified>.*)\]$") ?? suffix;
                        var purified = suffix.Extract(@"^\{(?<purified>.*)\}$");
                        suffix = purified ?? suffix;
                        Action<Mod> add = mod =>
                        {
                            if (purified == null) mandatory_mods.Add(mod);
                            else optional_mods.Add(mod);
                        };

                        if (suffix.StartsWith("|"))
                        {
                            add(Mod.Couple);
                        }
                        else if (suffix.StartsWith(".") && suffix.EndsWith("sel"))
                        {
                            add(Mod.B0);
                            add(Mod.B1);
                            add(Mod.B2);
                            add(Mod.B3);
                            add(Mod.H0);
                            add(Mod.H1);
                        }
                        else
                        {
                            throw AssertionHelper.Fail();
                        }
                    }

                    var p_operand = p_operands[ms.AssertSingle(kvp => kvp.Value != null).Key];
                    p_operands.RemoveElements(kvp => kvp.Value == p_operand);
                    operands.Add(new PtxopOperand(p_operand, p_operand.Name, optional_mods.ToReadOnly(), mandatory_mods.ToReadOnly()));
                }
                else
                {
                    throw AssertionHelper.Fail();
                }
            };

            var sig = meta.Signature;
            sig.ForEach((c, i) =>
            {
                var curr = sig.Slice(0, i);
                if (state == s_reading_opcode)
                {
                    if (quote != '\0')
                    {
                        throw AssertionHelper.Fail();
                    }
                    else
                    {
                        if (c == '{') { flush(); quote = c; state = s_reading_affix; }
                        else if (c == '}') throw AssertionHelper.Fail();
                        else if (c == '.') { flush(); state = s_reading_affix; }
                        else if (c == ',') throw AssertionHelper.Fail();
                        else if (c == ' ') { flush(); state = s_expecting_operand; }
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf.Append(c); state = s_reading_opcode; }
                    }
                }
                else if (state == s_expecting_affix)
                {
                    if (quote != '\0')
                    {
                        throw AssertionHelper.Fail();
                    }
                    else
                    {
                        if (c == '{') { quote = c; state = s_reading_affix; }
                        else if (c == '}') throw AssertionHelper.Fail();
                        else if (c == '.') { state = s_reading_affix; }
                        else if (c == ',') throw AssertionHelper.Fail();
                        else if (c == ' ') { state = s_expecting_operand; }
                        else if (c == ';') { state = s_finish; }
                        else { buf.Append(c); state = s_reading_affix; }
                    }
                }
                else if (state == s_reading_affix)
                {
                    if (quote != '\0')
                    {
                        if (c == '{') throw AssertionHelper.Fail();
                        else if (c == '}') { (quote == '{').AssertTrue(); flush(); quote = '\0'; state = s_expecting_affix; }
                        else if (c == '.') { state = s_reading_affix; }
                        else if (c == ';') throw AssertionHelper.Fail();
                        else { buf.Append(c); state = s_reading_affix; }
                    }
                    else
                    {
                        if (c == '{') { flush(); quote = c; state = s_reading_affix; }
                        else if (c == '}') throw AssertionHelper.Fail();
                        else if (c == '.') { flush(); state = s_expecting_affix; }
                        else if (c == ',') throw AssertionHelper.Fail();
                        else if (c == ' ') { flush(); state = s_expecting_operand; }
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf.Append(c); state = s_reading_affix; }
                    }
                }
                else if (state == s_expecting_operand)
                {
                    if (quote != '\0')
                    {
                        throw AssertionHelper.Fail();
                    }
                    else
                    {
                        if (c == '[') { state = s_expecting_operand; }
                        else if (c == ']') { state = s_expecting_operand; }
                        else if (c == '.') throw AssertionHelper.Fail();
                        else if (c == ',') throw AssertionHelper.Fail();
                        else if (c == ' ') { state = s_expecting_operand; }
                        else if (c == ';') { state = s_finish; }
                        else { buf.Append(c); state = s_reading_operand; }
                    }
                }
                else if (state == s_reading_operand)
                {
                    if (quote != '\0')
                    {
                        throw AssertionHelper.Fail();
                    }
                    else
                    {
                        if (c == '[') { state = s_reading_operand; }
                        else if (c == ']') { state = s_reading_operand; }
                        else if (c == ',') { flush(); state = s_expecting_operand; }
                        else if (c == ' ') { flush(); state = s_expecting_operand; }
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf.Append(c); state = s_reading_operand; }
                    }
                }
                else if (state == s_finish)
                {
                    throw AssertionHelper.Fail();
                }
            });
            (state == s_finish).AssertTrue();

            Mods = mods.ToReadOnly();
            Affixes = affixes.ToReadOnly();
            Operands = operands.ToReadOnly();
        }

        public override String ToString()
        {
            return Meta.Signature;
        }
    }
}