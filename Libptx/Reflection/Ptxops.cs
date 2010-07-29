using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Libptx.Instructions;
using Libptx.Instructions.Annotations;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Ptxops
    {
        private static readonly ReadOnlyCollection<Type> _cache;

        static Ptxops()
        {
            var libptx = typeof(ptxop).Assembly;
            _cache = libptx.GetTypes().Where(t => t.BaseType == typeof(ptxop)).ToReadOnly();
        }

        public static ReadOnlyCollection<Type> All
        {
            get { return _cache; }
        }

        public static ReadOnlyCollection<PtxopSig> PtxopSigs(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var t = obj as Type;
                if (t != null)
                {
                    return t.Attrs<PtxopAttribute>().Select(a => new PtxopSig(t, a)).ToReadOnly();
                }

                return obj.GetType().PtxopSigs();
            }
        }

        public static PtxopMeta PtxopMeta(this ptxop ptxop)
        {
            return ptxop == null ? null : new PtxopMeta(ptxop);
        }
    }
}