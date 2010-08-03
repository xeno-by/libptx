using System;
using System.Collections.Generic;
using System.Linq;
using Libptx.Common.Types;
using XenoGears.Assertions;
using XenoGears.Functional;
using XenoGears.Strings.Writers;
using Type = Libptx.Common.Types.Type;

namespace Libptx.Edsl.TextGenerators.Common
{
    internal static class Emitter
    {
        public static void EmitTypeSpec(this IndentedWriter w, String code_type, Type assumed_type, Space assumed_space)
        {
            w.EmitTypeSpec(code_type, assumed_type, assumed_space, false);
        }

        public static void EmitTypeSpec(this IndentedWriter w, String code_type, Type assumed_type, Space assumed_space, bool full_name)
        {
            assumed_type.AssertNotNull();
            if (assumed_type.is_arr()) return;

            // 1. Infer strict and relaxed conversions for element type

            var strict = new List<String>();
            var relaxed = new List<String>();

            var el = assumed_type;
            el = el.Unfold(t => t.arr_el(), t => t != null).Last();
            el = el.Unfold(t => t.vec_el(), t => t != null).Last();
            var s_sizes = (el.SizeInMemory * 8).MkArray();
            var r_sizes = (new[] { 8, 16, 32, 64 }).Where(sz1 => sz1 <= el.SizeInMemory * 8).Reverse().ToReadOnly();

            if (assumed_type.is_int())
            {
                s_sizes.ForEach(sz1 => strict.Add("u" + sz1));
                r_sizes.ForEach(sz1 => relaxed.Add("u" + sz1));
                s_sizes.ForEach(sz1 => strict.Add("s" + sz1));
                r_sizes.ForEach(sz1 => relaxed.Add("s" + sz1));
                s_sizes.ForEach(sz1 => strict.Add("b" + sz1));
                r_sizes.ForEach(sz1 => relaxed.Add("b" + sz1));
            }
            else if (assumed_type.is_float())
            {
                s_sizes.ForEach(sz1 => strict.Add("f" + sz1));
                r_sizes.ForEach(sz1 => relaxed.Add("f" + sz1));
                s_sizes.ForEach(sz1 => strict.Add("b" + sz1));
                r_sizes.ForEach(sz1 => relaxed.Add("b" + sz1));
            }
            else
            {
                assumed_type.is_bit().AssertTrue();

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

            // 2. Hack around conversions for usage with vectors

            if (assumed_type.is_vec())
            {
                strict.ForEach((t, i) => strict[i] = String.Format("v{0}_{1}", assumed_type.vec_rank(), t));
                relaxed.ForEach((t, i) => relaxed[i] = String.Format("v{0}_{1}", assumed_type.vec_rank(), t));
            }

            // 3. Emit appropriate implicit conversions
            var ns = "Libptx.Edsl.Common.Types.";
            if (assumed_type.is_opaque()) ns += "Opaque.";
            else if (assumed_type.is_vec()) ns += "Vector.";
            else ns += "Scalar.";
            if (!full_name) ns = null;

            if (assumed_space == Space.Reg)
            {
                strict.ForEach(t => w.WriteLine("public static implicit operator {2}reg_{0}({1} {1}) {{ return new {2}reg_{0}({1}); }}", t, code_type, ns));
                relaxed.ForEach(t => w.WriteLine("public static implicit operator {2}relaxed_reg_{0}({1} {1}) {{ return new {2}relaxed_reg_{0}({1}); }}", t, code_type, ns));
            }

            strict.ForEach(t => w.WriteLine("public static implicit operator {2}{0}({1} {1}) {{ return new {2}{0}({1}); }}", t, code_type, ns));
        }
    }
}
