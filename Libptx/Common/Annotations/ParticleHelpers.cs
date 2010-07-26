using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libcuda.Versions;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;
using System.Linq;
using XenoGears.Functional;

namespace Libptx.Common.Annotations
{
    [DebuggerNonUserCode]
    public static class ParticleHelpers
    {
        public static String Signature(this Object obj)
        {
            var sigs = obj.Signatures();
            return sigs == null ? null : sigs.SingleOrDefault();
        }

        public static ReadOnlyCollection<String> Signatures(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcls = cap.Attrs<ParticleAttribute>();
                    return pcls.Select(pcl => pcl.Signature).ToReadOnly();
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Signatures();
                }

                return t.Signatures();
            }
        }

        public static SoftwareIsa Version(this Object obj)
        {
            var sigs = obj.Versions();
            return sigs == null ? 0 : sigs.SingleOrDefault();
        }

        public static ReadOnlyCollection<SoftwareIsa> Versions(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcls = cap.Attrs<ParticleAttribute>();
                    return pcls.Select(pcl => pcl.Version).ToReadOnly();
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Versions();
                }

                return t.Versions();
            }
        }

        public static HardwareIsa Target(this Object obj)
        {
            var sigs = obj.Targets();
            return sigs == null ? 0 : sigs.SingleOrDefault();
        }

        public static ReadOnlyCollection<HardwareIsa> Targets(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcls = cap.Attrs<ParticleAttribute>();
                    return pcls.Select(pcl => pcl.Target).ToReadOnly();
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Targets();
                }

                return t.Targets();
            }
        }
    }
}