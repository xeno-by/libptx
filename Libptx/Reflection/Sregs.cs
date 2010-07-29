using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Libptx.Expressions.Sregs;
using System.Linq;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Sregs
    {
        private static readonly ReadOnlyCollection<Type> _cache;

        static Sregs()
        {
            var libptx = typeof(Sreg).Assembly;
            _cache = libptx.GetTypes().Where(t => t.BaseType == typeof(Sreg)).ToReadOnly();
        }

        public static ReadOnlyCollection<Type> All
        {
            get { return _cache; }
        }

        public static ReadOnlyCollection<SregSig> Sigs
        {
            get { return _cache.Select(t => t.SregSig()).ToReadOnly(); }
        }

        public static SregSig SregSig(this Object obj)
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
                    var a = t.AttrOrNull<SregAttribute>();
                    return a != null ? new SregSig(t, a) : null;
                }

                return obj.GetType().SregSig();
            }
        }
    }
}