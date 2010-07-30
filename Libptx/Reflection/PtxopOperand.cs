using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libptx.Expressions;

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
        {
            Decl = decl;
            Name = name;
        }

        internal PtxopOperand(PropertyInfo decl, String name, ReadOnlyCollection<Mod> optionalMods, ReadOnlyCollection<Mod> mandatoryMods)
        {
            Decl = decl;
            Name = name;
            OptionalMods = optionalMods;
            MandatoryMods = mandatoryMods;
        }
    }
}