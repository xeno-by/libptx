using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libptx.Common.Annotations.Quanta;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopMod
    {
        public PropertyInfo Decl { get; private set; }
        public ModAttribute Meta { get; private set; }

        public String Name { get; private set; }
        public bool IsMandatory { get; private set; }
        public ReadOnlyCollection<Object> Options { get; private set; }

        internal PtxopMod(PropertyInfo decl, ModAttribute meta, String name, bool isMandatory)
        {
            Decl = decl;
            Meta = meta;
            Name = name;
            IsMandatory = isMandatory;
        }

        internal PtxopMod(PropertyInfo decl, ModAttribute meta, String name, bool isMandatory, ReadOnlyCollection<Object> options)
        {
            Decl = decl;
            Meta = meta;
            Name = name;
            IsMandatory = isMandatory;
            Options = options;
        }
    }
}