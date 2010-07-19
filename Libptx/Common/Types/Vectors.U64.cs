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
    public struct ulong3 : IEnumerable<ulong>, IEquatable<ulong3>
    {
        public ulong X;
        public ulong Y;
        public ulong Z;

        public ulong3(ulong x) : this(x, default(ulong), default(ulong)) { }
        public ulong3(ulong x, ulong y) : this(x, y, default(ulong)) { }
        public ulong3(ulong x, ulong y, ulong z) { X = x; Y = y; Z = z; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<ulong> GetEnumerator() { return new[] { X, Y, Z }.Cast<ulong>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(ulong).Name, String.Format("({0}, {1}, {2})", X, Y, Z)); }

        public bool Equals(ulong3 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(ulong3)) return false;
            return Equals((ulong3)obj);
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

        public static bool operator ==(ulong3 left, ulong3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(ulong3 left, ulong3 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct ulong4 : IEnumerable<ulong>, IEquatable<ulong4>
    {
        public ulong X;
        public ulong Y;
        public ulong Z;
        public ulong W;

        public ulong4(ulong x) : this(x, default(ulong), default(ulong), default(ulong)) { }
        public ulong4(ulong x, ulong y) : this(x, y, default(ulong), default(ulong)) { }
        public ulong4(ulong x, ulong y, ulong z) : this(x, y, z, default(ulong)) { }
        public ulong4(ulong x, ulong y, ulong z, ulong w) { X = x; Y = y; Z = z; W = w; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<ulong> GetEnumerator() { return new[] { X, Y, Z, W }.Cast<ulong>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(ulong).Name, String.Format("({0}, {1}, {2}, {3})", X, Y, Z, W)); }

        public bool Equals(ulong4 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z) && Equals(other.W, W);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(ulong4)) return false;
            return Equals((ulong4)obj);
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

        public static bool operator ==(ulong4 left, ulong4 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(ulong4 left, ulong4 right)
        {
            return !left.Equals(right);
        }
    }
}