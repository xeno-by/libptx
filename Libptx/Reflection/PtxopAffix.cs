using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libptx.Common.Annotations.Quanta;

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
        {
            Decl = decl;
            Meta = meta;
            Name = name;
            IsMandatory = isMandatory;
        }

        internal PtxopAffix(PropertyInfo decl, AffixAttribute meta, String name, bool isMandatory, ReadOnlyCollection<Object> options)
        {
            Decl = decl;
            Meta = meta;
            Name = name;
            IsMandatory = isMandatory;
            Options = options;
        }
    }
}