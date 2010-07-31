using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using System.Text;
using Libptx.Common.Annotations.Quanta;
using XenoGears.Functional;
using System.Linq;
using XenoGears.Strings;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopAffix
    {
        public PropertyInfo Decl { get; private set; }
        public AffixAttribute Meta { get; private set; }

        public String Name { get; private set; }
        public bool IsMandatory { get; private set; }
        public ReadOnlyCollection<Object> Options { get; private set; }

        internal PtxopAffix(PropertyInfo decl, AffixAttribute meta, String name, bool isMandatory)
            : this(decl, meta, name, isMandatory, null)
        {
        }

        internal PtxopAffix(PropertyInfo decl, AffixAttribute meta, String name, bool isMandatory, ReadOnlyCollection<Object> options)
        {
            Decl = decl;
            Meta = meta;
            Name = name;
            IsMandatory = isMandatory;
            Options = options ?? new ReadOnlyCollection<Object>(new List<Object>());
        }

        public override String ToString()
        {
            var buf = new StringBuilder();
            buf.AppendFormat("{0}{1}", IsMandatory ? "+" : "?", Name);
            if (Options.IsNotEmpty()) buf.AppendFormat(" = [{0}]", Options.Select(opt => opt.Signature() ?? opt.ToInvariantString()).StringJoin());
            return buf.ToString();
        }
    }
}