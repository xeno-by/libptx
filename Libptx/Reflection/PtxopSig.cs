using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libptx.Common.Annotations.Quanta;
using Libptx.Instructions.Annotations;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class PtxopSig
    {
        public Type Decl { get; private set; }
        public PtxopAttribute Meta { get; private set; }

        public String Opcode { get; private set; }
        public ReadOnlyCollection<PtxopAffix> Mods { get; private set; }
        public ReadOnlyCollection<PtxopAffix> Affixes { get; private set; }
        public ReadOnlyCollection<PtxopOperand> Operands { get; private set; }

        internal PtxopSig(Type decl, PtxopAttribute meta)
        {
            Decl = decl;
            Meta = meta;

            throw new NotImplementedException();
        }
    }

    [DebuggerNonUserCode]
    public class PtxopMod
    {
        public PropertyInfo Decl { get; private set; }
        public ModAttribute Meta { get; private set; }

        public String Name { get; private set; }
        public bool IsMandatory { get; private set; }
        public ReadOnlyCollection<Object> Options { get; private set; }
    }

    [DebuggerNonUserCode]
    public class PtxopAffix
    {
        public PropertyInfo Decl { get; private set; }
        public AffixAttribute Meta { get; private set; }

        public String Name { get; private set; }
        public bool IsMandatory { get; private set; }
        public ReadOnlyCollection<Object> Options { get; private set; }
    }

    [DebuggerNonUserCode]
    public class PtxopOperand
    {
        public PropertyInfo Decl { get; private set; }
        public String Name { get; private set; }
    }
}