using System;
using System.Reflection;
using Libcuda.Versions;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;
using System.Linq;

namespace Libptx.Common.Annotations
{
    public static class ParticleHelpers
    {
        public static String Signature(this Object obj)
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
                    var pcl = cap.AttrOrNull<ParticleAttribute>();
                    return pcl == null ? null : pcl.Signature;
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Signature();
                }

                return t.Signature();
            }
        }

        public static SoftwareIsa Version(this Object obj)
        {
            if (obj == null)
            {
                return 0;
            }
            else
            {
                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcl = cap.AttrOrNull<ParticleAttribute>();
                    return pcl == null ? 0 : pcl.Version;
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Version();
                }

                return t.Version();
            }
        }

        public static HardwareIsa Target(this Object obj)
        {
            if (obj == null)
            {
                return 0;
            }
            else
            {
                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcl = cap.AttrOrNull<ParticleAttribute>();
                    return pcl == null ? 0 : pcl.Target;
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).SingleOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Target();
                }

                return t.Target();
            }
        }
    }
}