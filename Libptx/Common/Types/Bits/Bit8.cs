using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace Libptx.Common.Types.Bits
{
    // todo. convenient inits for bX constants (just gief implicit/explicit casts for scalar and vector versions)

    [DebuggerNonUserCode]
    public struct Bit8
    {
        public byte Raw { get; set; }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit8_V1 : IEnumerable<Bit8>, IEquatable<Bit8_V1>
    {
        public Bit8 X;
        public Bit8_V1(Bit8 x) { X = x; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit8> GetEnumerator() { return new[] { X }.Cast<Bit8>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit8).Name, String.Format("({0})", X)); }

        public bool Equals(Bit8_V1 other)
        {
            return Equals(other.X, X);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit8_V1)) return false;
            return Equals((Bit8_V1)obj);
        }

        public override int GetHashCode()
        {
            return X.GetHashCode();
        }

        public static bool operator ==(Bit8_V1 left, Bit8_V1 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit8_V1 left, Bit8_V1 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit8_V2 : IEnumerable<Bit8>, IEquatable<Bit8_V2>
    {
        public Bit8 X;
        public Bit8 Y;

        public Bit8_V2(Bit8 x) : this(x, default(Bit8)) { }
        public Bit8_V2(Bit8 x, Bit8 y) { X = x; Y = y; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit8> GetEnumerator() { return new[] { X, Y }.Cast<Bit8>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit8).Name, String.Format("({0}, {1})", X, Y)); }

        public bool Equals(Bit8_V2 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit8_V2)) return false;
            return Equals((Bit8_V2)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (X.GetHashCode() * 397) ^ Y.GetHashCode();
            }
        }

        public static bool operator ==(Bit8_V2 left, Bit8_V2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit8_V2 left, Bit8_V2 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit8_V3 : IEnumerable<Bit8>, IEquatable<Bit8_V3>
    {
        public Bit8 X;
        public Bit8 Y;
        public Bit8 Z;

        public Bit8_V3(Bit8 x) : this(x, default(Bit8), default(Bit8)) { }
        public Bit8_V3(Bit8 x, Bit8 y) : this(x, y, default(Bit8)) { }
        public Bit8_V3(Bit8 x, Bit8 y, Bit8 z) { X = x; Y = y; Z = z; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit8> GetEnumerator() { return new[] { X, Y, Z }.Cast<Bit8>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit8).Name, String.Format("({0}, {1}, {2})", X, Y, Z)); }

        public bool Equals(Bit8_V3 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit8_V3)) return false;
            return Equals((Bit8_V3)obj);
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

        public static bool operator ==(Bit8_V3 left, Bit8_V3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit8_V3 left, Bit8_V3 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit8_V4 : IEnumerable<Bit8>, IEquatable<Bit8_V4>
    {
        public Bit8 X;
        public Bit8 Y;
        public Bit8 Z;
        public Bit8 W;

        public Bit8_V4(Bit8 x) : this(x, default(Bit8), default(Bit8), default(Bit8)) { }
        public Bit8_V4(Bit8 x, Bit8 y) : this(x, y, default(Bit8), default(Bit8)) { }
        public Bit8_V4(Bit8 x, Bit8 y, Bit8 z) : this(x, y, z, default(Bit8)) { }
        public Bit8_V4(Bit8 x, Bit8 y, Bit8 z, Bit8 w) { X = x; Y = y; Z = z; W = w; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit8> GetEnumerator() { return new[] { X, Y, Z, W }.Cast<Bit8>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit8).Name, String.Format("({0}, {1}, {2}, {3})", X, Y, Z, W)); }

        public bool Equals(Bit8_V4 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z) && Equals(other.W, W);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit8_V4)) return false;
            return Equals((Bit8_V4)obj);
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

        public static bool operator ==(Bit8_V4 left, Bit8_V4 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit8_V4 left, Bit8_V4 right)
        {
            return !left.Equals(right);
        }
    }
}