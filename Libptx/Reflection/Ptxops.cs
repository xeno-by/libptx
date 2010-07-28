using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
using Libptx.Instructions;
using XenoGears.Functional;

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

        public static ReadOnlyCollection<String> Sigs
        {
            get { return _cache.SelectMany(t => t.Signatures()).ToReadOnly(); }
        }
    }
}