using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Reflection;
using Libcuda.Versions;
using Libptx.Common;
using Libptx.Common.Annotations;
using XenoGears.Reflection.Attributes;
using XenoGears.Reflection.Shortcuts;
using System.Linq;
using XenoGears.Functional;

namespace Libptx.Reflection
{
    [DebuggerNonUserCode]
    public static class ParticleReflector
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

        public static SoftwareIsa EigenVersion(this Object obj)
        {
            var atom = obj as Atom;
            if (atom != null)
            {
                return atom.EigenVersion;
            }
            else
            {
                var particles = obj.Particles();
                if (particles == null) return 0;
                return particles.Select(pcl => pcl.Version).Distinct().SingleOrDefault();
            }
        }

        public static SoftwareIsa VersionRecursive(this Object obj)
        {
            throw new NotImplementedException();
        }

        public static HardwareIsa EigenTarget(this Object obj)
        {
            var atom = obj as Atom;
            if (atom != null)
            {
                return atom.EigenTarget;
            }
            else
            {
                var particles = obj.Particles();
                if (particles == null) return 0;
                return particles.Select(pcl => pcl.Target).Distinct().SingleOrDefault();
            }
        }

        public static SoftwareIsa TargetRecursive(this Object obj)
        {
            throw new NotImplementedException();
        }
    }
}