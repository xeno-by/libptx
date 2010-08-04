using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using Libptx.Expressions.Sregs;
using System.Linq;
using Libptx.Expressions.Sregs.Annotations;
using XenoGears.Collections.Dictionaries;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Sregs
    {
        private static readonly ReadOnlyCollection<Type> _sregs;
        private static readonly ReadOnlyDictionary<Type, SregSig> _sigs;

        static Sregs()
        {
            var libptx = typeof(Sreg).Assembly;
            _sregs = libptx.GetTypes().Where(t => t.BaseType == typeof(Sreg)).OrderBy(t => t.Name).ToReadOnly();
            _sigs = _sregs.ToDictionary(t => t, t => new SregSig(t, t.Attr<SregAttribute>())).ToReadOnly();
        }

        public static ReadOnlyCollection<Type> All
        {
            get { return _sregs; }
        }

        public static ReadOnlyCollection<SregSig> Sigs
        {
            get { return _sigs.Values.ToReadOnly(); }
        }

        public static SregSig SregSig(this Object obj)
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
    }
}