using System;
using System.Diagnostics;
using Libcuda.Versions;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Strings;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public class SregSig
    {
        public Type Decl { get; private set; }
        public SregAttribute Meta { get; private set; }

        public SoftwareIsa Version { get { return Meta.Version; } }
        public HardwareIsa Target { get { return Meta.Target; } }

        public String Name { get; private set; }
        public Type Type { get; private set; }

        internal SregSig(Type decl, SregAttribute meta)
        {
            Decl = decl;
            Meta = meta;

            Name = meta.Signature ?? decl.Name;
            Type = meta.Type;
        }

        public override String ToString()
        {
            return String.Format("{0} of type {1}", Name, Type.GetCSharpRef(ToCSharpOptions.Terse));
        }
    }
}