using System;
using System.Diagnostics;
using System.Linq;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Annotations.Quanta;
using XenoGears;
using XenoGears.Functional;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    internal static class Atoms
    {
        public static SoftwareIsa EigenVersion(this Atom atom)
        {
            if (atom == null) return 0;
            var t_swisa = atom.GetType().Version();

            var props = atom.GetType().GetProperties(BF.PublicInstance).Where(p => p.HasAttr<QuantumAttribute>()).ToReadOnly();
            var props_swisa = props.Select(p =>
            {
                var v = p.GetValue(atom, null);
                var v_swisa = v.Version();

                var @default = p.PropertyType.Fluent(t => t.IsValueType ? Activator.CreateInstance(t) : null);
                var p_swisa = Equals(v, @default) ? 0 : p.Version();

                return (SoftwareIsa)Math.Max((int)v_swisa, (int)p_swisa);
            }).MaxOrDefault();

            return (SoftwareIsa)Math.Max((int)t_swisa, (int)props_swisa);
        }

        public static HardwareIsa EigenTarget(this Atom atom)
        {
            if (atom == null) return 0;
            var t_hwisa = atom.GetType().Target();

            var props = atom.GetType().GetProperties(BF.PublicInstance).Where(p => p.HasAttr<QuantumAttribute>()).ToReadOnly();
            var props_hwisa = props.Select(p =>
            {
                var v = p.GetValue(atom, null);
                var v_hwisa = v.Target();

                var @default = p.PropertyType.Fluent(t => t.IsValueType ? Activator.CreateInstance(t) : null);
                var p_hwisa = Equals(v, @default) ? 0 : p.Target();

                return (HardwareIsa)Math.Max((int)v_hwisa, (int)p_hwisa);
            }).MaxOrDefault();

            return (HardwareIsa)Math.Max((int)t_hwisa, (int)props_hwisa);
        }
    }
}
