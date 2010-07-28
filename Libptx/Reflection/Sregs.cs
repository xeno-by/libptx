using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Libptx.Expressions.Sregs;
using System.Linq;
using XenoGears.Functional;

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

        public static ReadOnlyCollection<String> Sigs
        {
            get { return _cache.Select(t => t.Signature()).ToReadOnly(); }
        }
    }
}