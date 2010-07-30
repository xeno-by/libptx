using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using Libptx.Instructions;
using Libptx.Instructions.Annotations;
using XenoGears.Collections.Dictionaries;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Ptxops
    {
        private static readonly ReadOnlyCollection<Type> _ptxops;
        private static readonly ReadOnlyDictionary<Type, ReadOnlyCollection<PtxopSig>> _sigs;

        static Ptxops()
        {
            var libptx = typeof(ptxop).Assembly;
            _ptxops = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();
            _sigs = _ptxops.ToDictionary(t => t, t => t.Attrs<PtxopAttribute>().Select(a => new PtxopSig(t, a)).ToReadOnly()).ToReadOnly();
        }

        public static ReadOnlyCollection<Type> All
        {
            get { return _ptxops; }
        }

        public static ReadOnlyCollection<PtxopSig> PtxopSigs(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var t = obj as Type ?? obj.GetType();
                return _sigs.GetOrDefault(t);
            }
        }

        public static PtxopState PtxopState(this ptxop ptxop)
        {
            return ptxop == null ? null : new PtxopState(ptxop);
        }
    }
}