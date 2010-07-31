using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using Libptx.Expressions;
using XenoGears.Functional;
using System.Linq;
using XenoGears.Strings;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopOperand
    {
        public PropertyInfo Decl { get; private set; }
        public String Name { get; private set; }

        public ReadOnlyCollection<Mod> OptionalMods { get; private set; }
        public ReadOnlyCollection<Mod> MandatoryMods { get; private set; }

        internal PtxopOperand(PropertyInfo decl, String name)
            : this(decl, name, null, null)
        {
        }

        internal PtxopOperand(PropertyInfo decl, String name, ReadOnlyCollection<Mod> optionalMods, ReadOnlyCollection<Mod> mandatoryMods)
        {
            Decl = decl;
            Name = name;
            OptionalMods = optionalMods ?? new ReadOnlyCollection<Mod>(new List<Mod>());
            MandatoryMods = mandatoryMods ?? new ReadOnlyCollection<Mod>(new List<Mod>());
        }

        public override String ToString()
        {
            var buf = new StringBuilder();
            buf.Append(Name);

            if (OptionalMods.IsNotEmpty() || MandatoryMods.IsNotEmpty())
            {
                var s_oms = OptionalMods.Select(om => String.Format("?{0}", om.Signature() ?? om.ToInvariantString()));
                var s_mms = MandatoryMods.Select(mm => String.Format("+{0}", mm.Signature() ?? mm.ToInvariantString()));
                buf.AppendFormat(", mods = [{0}]", s_oms.Concat(s_mms).StringJoin());
            }

            return buf.ToString();
        }
    }
}