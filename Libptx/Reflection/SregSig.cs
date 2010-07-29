using System;
using System.Diagnostics;
using Libptx.Expressions.Sregs.Annotations;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class SregSig
    {
        public Type Decl { get; private set; }
        public SregAttribute Meta { get; private set; }

        public String Name { get; private set; }
        public Type Type { get; private set; }

        internal SregSig(Type decl, SregAttribute meta)
        {
            Decl = decl;
            Meta = meta;

            Name = meta.Signature;
            Type = meta.Type;
        }
    }
}