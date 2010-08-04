using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Libptx.Common.Types;
using Libptx.Expressions;
using Libptx.Expressions.Addresses;
using Libptx.Expressions.Slots;
using Libptx.Functions;
using XenoGears.Assertions;
using XenoGears.Functional;

namespace Libptx.Common.Names
{
    [DebuggerNonUserCode]
    internal static class Namer
    {
        private static HashSet<String> _names = new HashSet<String>();

        public static String GenName(this Slot s)
        {
            Func<String> prefix = () =>
            {
                var s_reg = s as Reg;
                if (s_reg != null)
                {
                    Func<String> basis = () =>
                    {
                        switch (s_reg.Type.Name)
                        {
                            case TypeName.U8:
                                return "rhh";
                            case TypeName.S8:
                                return "rhhs";
                            case TypeName.U16:
                                return "rhh";
                            case TypeName.S16:
                                return "rhs";
                            case TypeName.U32:
                                return "r";
                            case TypeName.S32:
                                return "rs";
                            case TypeName.U64:
                                return "rd";
                            case TypeName.S64:
                                return "rds";
                            case TypeName.F16:
                                return "fh";
                            case TypeName.F32:
                                return "f";
                            case TypeName.F64:
                                return "fd";
                            case TypeName.B8:
                                return "bhh";
                            case TypeName.B16:
                                return "bh";
                            case TypeName.B32:
                                return "b";
                            case TypeName.B64:
                                return "bd";
                            case TypeName.Pred:
                                return "p";
                            default:
                                throw AssertionHelper.Fail();
                        }
                    };

                    String vec = null;
                    if (!s_reg.is_vec()) vec = String.Empty;
                    else if (s_reg.is_v1()) vec = "v1_";
                    else if (s_reg.is_v2()) vec = "v2_";
                    else if (s_reg.is_v4()) vec = "v4_";
                    else throw AssertionHelper.Fail();

                    s_reg.is_arr().AssertFalse();
                    return vec + basis();
                }

                var s_var = s as Var;
                if (s_var != null) return "var";

                throw AssertionHelper.Fail();
            };

            Func<int, String> gen = i => String.Format("%{0}{1}", prefix(), i);
            var gend_name = Seq.Nats.Skip(1).Select(gen).First(name => !_names.Contains(name));
            _names.Add(gend_name);
            return gend_name;
        }

        public static String GenName(this Label lbl)
        {
            Func<int, String> gen = i => String.Format("%{0}{1}", "lbl", i);
            var gend_name = Seq.Nats.Skip(1).Select(gen).First(name => !_names.Contains(name));
            _names.Add(gend_name);
            return gend_name;
        }

        public static String GenName(this Entry entry)
        {
            Func<int, String> gen = i => String.Format("%{0}{1}", "entry", i);
            var gend_name = Seq.Nats.Skip(1).Select(gen).First(name => !_names.Contains(name));
            _names.Add(gend_name);
            return gend_name;
        }
    }
}