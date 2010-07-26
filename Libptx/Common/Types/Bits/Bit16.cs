using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace Libptx.Common.Types.Bits
{
    [DebuggerNonUserCode]
    public struct Bit16
    {
        internal short _fillerForSizeof;
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit16_V1 : IEnumerable<Bit16>, IEquatable<Bit16_V1>
    {
        public Bit16 X;
        public Bit16_V1(Bit16 x) { X = x; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit16> GetEnumerator() { return new[] { X }.Cast<Bit16>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit16).Name, String.Format("({0})", X)); }

        public bool Equals(Bit16_V1 other)
        {
            return Equals(other.X, X);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit16_V1)) return false;
            return Equals((Bit16_V1)obj);
        }

        public override int GetHashCode()
        {
            return X.GetHashCode();
        }

        public static bool operator ==(Bit16_V1 left, Bit16_V1 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit16_V1 left, Bit16_V1 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit16_V2 : IEnumerable<Bit16>, IEquatable<Bit16_V2>
    {
        public Bit16 X;
        public Bit16 Y;

        public Bit16_V2(Bit16 x) : this(x, default(Bit16)) { }
        public Bit16_V2(Bit16 x, Bit16 y) { X = x; Y = y; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit16> GetEnumerator() { return new[] { X, Y }.Cast<Bit16>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit16).Name, String.Format("({0}, {1})", X, Y)); }

        public bool Equals(Bit16_V2 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit16_V2)) return false;
            return Equals((Bit16_V2)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (X.GetHashCode() * 397) ^ Y.GetHashCode();
            }
        }

        public static bool operator ==(Bit16_V2 left, Bit16_V2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit16_V2 left, Bit16_V2 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit16_V3 : IEnumerable<Bit16>, IEquatable<Bit16_V3>
    {
        public Bit16 X;
        public Bit16 Y;
        public Bit16 Z;

        public Bit16_V3(Bit16 x) : this(x, default(Bit16), default(Bit16)) { }
        public Bit16_V3(Bit16 x, Bit16 y) : this(x, y, default(Bit16)) { }
        public Bit16_V3(Bit16 x, Bit16 y, Bit16 z) { X = x; Y = y; Z = z; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit16> GetEnumerator() { return new[] { X, Y, Z }.Cast<Bit16>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit16).Name, String.Format("({0}, {1}, {2})", X, Y, Z)); }

        public bool Equals(Bit16_V3 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit16_V3)) return false;
            return Equals((Bit16_V3)obj);
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

        public static bool operator ==(Bit16_V3 left, Bit16_V3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit16_V3 left, Bit16_V3 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit16_V4 : IEnumerable<Bit16>, IEquatable<Bit16_V4>
    {
        public Bit16 X;
        public Bit16 Y;
        public Bit16 Z;
        public Bit16 W;

        public Bit16_V4(Bit16 x) : this(x, default(Bit16), default(Bit16), default(Bit16)) { }
        public Bit16_V4(Bit16 x, Bit16 y) : this(x, y, default(Bit16), default(Bit16)) { }
        public Bit16_V4(Bit16 x, Bit16 y, Bit16 z) : this(x, y, z, default(Bit16)) { }
        public Bit16_V4(Bit16 x, Bit16 y, Bit16 z, Bit16 w) { X = x; Y = y; Z = z; W = w; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit16> GetEnumerator() { return new[] { X, Y, Z, W }.Cast<Bit16>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit16).Name, String.Format("({0}, {1}, {2}, {3})", X, Y, Z, W)); }

        public bool Equals(Bit16_V4 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z) && Equals(other.W, W);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit16_V4)) return false;
            return Equals((Bit16_V4)obj);
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

        public static bool operator ==(Bit16_V4 left, Bit16_V4 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit16_V4 left, Bit16_V4 right)
        {
            return !left.Equals(right);
        }
    }
}