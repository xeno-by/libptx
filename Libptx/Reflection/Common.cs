using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libcuda.Versions;
using Libptx.Common.Annotations;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;
using System.Linq;
using XenoGears.Functional;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class Common
    {
        public static ParticleAttribute Particle(this Object obj)
        {
            var particles = obj.Particles();
            return particles == null ? null : particles.SingleOrDefault();
        }

        public static ReadOnlyCollection<ParticleAttribute> Particles(this Object obj)
        {
            if (obj == null)
            {
                return null;
            }
            else
            {
                var pcla = obj as ParticleAttribute;
                if (pcla != null)
                {
                    return pcla.MkArray().ToReadOnly();
                }

                var cap = obj as ICustomAttributeProvider;
                if (cap != null)
                {
                    var pcls = cap.Attrs<ParticleAttribute>();
                    return pcls.ToReadOnly();
                }

                var t = obj.GetType();
                if (t.IsEnum)
                {
                    var f = t.GetFields(BF.PublicStatic).FirstOrDefault(f1 => Equals(f1.GetValue(null), obj));
                    return f.Particles();
                }

                return t.Particles();
            }
        }

        public static String Signature(this Object obj)
        {
            var signatures = obj.Signatures();
            if (signatures == null) return null;
            return signatures.Distinct().SingleOrDefault();
        }

        public static ReadOnlyCollection<String> Signatures(this Object obj)
        {
            var particles = obj.Particles();
            return particles == null ? null : particles.Select(pcl => pcl.Signature).ToReadOnly();
        }

        public static SoftwareIsa Version(this Object obj)
        {
            var versions = obj.Versions();
            if (versions == null) return 0;
            return versions.Distinct().SingleOrDefault();
        }

        public static ReadOnlyCollection<SoftwareIsa> Versions(this Object obj)
        {
            var particles = obj.Particles();
            return particles == null ? null : particles.Select(pcl => pcl.Version).ToReadOnly();
        }

        public static HardwareIsa Target(this Object obj)
        {
            var targets = obj.Targets();
            if (targets == null) return 0;
            return targets.Distinct().SingleOrDefault();
        }

        public static ReadOnlyCollection<HardwareIsa> Targets(this Object obj)
        {
            var particles = obj.Particles();
            return particles == null ? null : particles.Select(pcl => pcl.Target).ToReadOnly();
        }
    }
}