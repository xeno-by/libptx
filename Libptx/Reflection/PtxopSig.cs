using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Text;
using Libcuda.Versions;
using Libptx.Common.Annotations.Quanta;
using Libptx.Expressions;
using Libptx.Instructions;
using Libptx.Instructions.Annotations;
using XenoGears.Assertions;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;

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

            var props = decl.GetType().GetProperties(BF.PublicInstance);
            var p_mods = props.Where(p => p.HasAttr<ModAttribute>()).ToReadOnly();
            var p_affixes = props.Where(p => p.HasAttr<AffixAttribute>()).ToReadOnly();
            var p_operands = props.Where(p => typeof(Expression).IsAssignableFrom(p.PropertyType)).ToReadOnly();

            var mods = new List<PtxopMod>();
            var affixes = new List<PtxopAffix>();
            var operands = new List<PtxopOperand>();
            int s_reading_opcode = 0, s_expecting_affix = 1, s_reading_affix = 2, s_expecting_operand = 3, s_reading_operand = 4, s_finish = 5;
            var state = s_reading_opcode;
            var buf = new StringBuilder();
            Action flush = () =>
            {
                if (state == s_reading_opcode)
                {
                    // todo. validate symbols
                    throw new NotImplementedException();
                }
                else if (state == s_reading_affix)
                {
                    // todo. validate symbols
                    // parse "{hi, lo}" and "u32"
                    throw new NotImplementedException();
                }
                else if (state == s_reading_operand)
                {
                    // todo. validate symbols
                    // parse "{-}d.dsel"
                    throw new NotImplementedException();
                }
                else
                {
                    throw AssertionHelper.Fail();
                }
            };

            char quote = '\0';
            meta.Signature.EndsWith(";").AssertTrue();
            meta.Signature.ForEach(c =>
            {
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
                        else if (c == '[') { flush(); quote = c; state = s_reading_affix; }
                        else if (c == ']') throw AssertionHelper.Fail();
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
                        if (c == '{') { flush(); quote = c; state = s_reading_affix; }
                        else if (c == '}') throw AssertionHelper.Fail();
                        else if (c == '[') { flush(); quote = c; state = s_reading_affix; }
                        else if (c == ']') throw AssertionHelper.Fail();
                        else if (c == '.') { flush(); state = s_reading_affix; }
                        else if (c == ',') throw AssertionHelper.Fail();
                        else if (c == ' ') { flush(); state = s_expecting_operand; }
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf = new StringBuilder(); buf.Append(c); state = s_reading_affix; }
                    }
                }
                else if (state == s_reading_affix)
                {
                    if (quote != '\0')
                    {
                        if (c == '{') throw AssertionHelper.Fail();
                        else if (c == '}') { (quote == '}').AssertTrue(); flush(); quote = '\0'; state = s_expecting_affix; }
                        else if (c == '[') throw AssertionHelper.Fail();
                        else if (c == ']') { (quote == '[').AssertTrue(); flush(); quote = '\0'; state = s_expecting_affix; }
                        else if (c == '.') { state = s_reading_affix; }
                        else if (c == ';') throw AssertionHelper.Fail();
                        else { buf.Append(c); state = s_reading_affix; }
                    }
                    else
                    {
                        if (c == '{') throw AssertionHelper.Fail();
                        else if (c == '}') throw AssertionHelper.Fail();
                        else if (c == '[') throw AssertionHelper.Fail();
                        else if (c == ']') throw AssertionHelper.Fail();
                        else if (c == '.') { flush(); state = s_reading_affix; }
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
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf = new StringBuilder(); buf.Append(c); state = s_reading_operand; }
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
                        if (c == '[') { state = s_expecting_operand; }
                        else if (c == ']') { state = s_expecting_operand; }
                        else if (c == ',') { flush(); state = s_expecting_operand; }
                        else if (c == ' ') { flush(); state = s_expecting_operand; }
                        else if (c == ';') { flush(); state = s_finish; }
                        else { buf = new StringBuilder(); buf.Append(c); state = s_reading_operand; }
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
    }
}