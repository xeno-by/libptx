using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace Libptx.Common.Types
{
    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct long3 : IEnumerable<long>, IEquatable<long3>
    {
        public long X;
        public long Y;
        public long Z;

        public long3(long x) : this(x, default(long), default(long)) { }
        public long3(long x, long y) : this(x, y, default(long)) { }
        public long3(long x, long y, long z) { X = x; Y = y; Z = z; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<long> GetEnumerator() { return new[] { X, Y, Z }.Cast<long>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(long).Name, String.Format("({0}, {1}, {2})", X, Y, Z)); }

        public bool Equals(long3 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(long3)) return false;
            return Equals((long3)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int result = X.GetHashCode();
                result = (result * 397) ^ Y.GetHashCode();
                result = (result * 397) ^ Z.GetHashCode();
                return result;
            }
        }

        public static bool operator ==(long3 left, long3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(long3 left, long3 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct long4 : IEnumerable<long>, IEquatable<long4>
    {
        public long X;
        public long Y;
        public long Z;
        public long W;

        public long4(long x) : this(x, default(long), default(long), default(long)) { }
        public long4(long x, long y) : this(x, y, default(long), default(long)) { }
        public long4(long x, long y, long z) : this(x, y, z, default(long)) { }
        public long4(long x, long y, long z, long w) { X = x; Y = y; Z = z; W = w; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<long> GetEnumerator() { return new[] { X, Y, Z, W }.Cast<long>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(long).Name, String.Format("({0}, {1}, {2}, {3})", X, Y, Z, W)); }

        public bool Equals(long4 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z) && Equals(other.W, W);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(long4)) return false;
            return Equals((long4)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int result = X.GetHashCode();
                result = (result * 397) ^ Y.GetHashCode();
                result = (result * 397) ^ Z.GetHashCode();
                result = (result * 397) ^ W.GetHashCode();
                return result;
            }
        }

        public static bool operator ==(long4 left, long4 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(long4 left, long4 right)
        {
            return !left.Equals(right);
        }
    }
}