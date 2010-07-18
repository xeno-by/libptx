using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace Libptx.Common.Types
{
    public struct Bit64
    {
        internal long _fillerForSizeof;
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit64_V1 : IEnumerable<Bit64>, IEquatable<Bit64_V1>
    {
        public Bit64 X;
        public Bit64_V1(Bit64 x) { X = x; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit64> GetEnumerator() { return new[] { X }.Cast<Bit64>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit64).Name, String.Format("({0})", X)); }

        public bool Equals(Bit64_V1 other)
        {
            return Equals(other.X, X);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit64_V1)) return false;
            return Equals((Bit64_V1)obj);
        }

        public override int GetHashCode()
        {
            return X.GetHashCode();
        }

        public static bool operator ==(Bit64_V1 left, Bit64_V1 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit64_V1 left, Bit64_V1 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct Bit64_V2 : IEnumerable<Bit64>, IEquatable<Bit64_V2>
    {
        public Bit64 X;
        public Bit64 Y;

        public Bit64_V2(Bit64 x) : this(x, default(Bit64)) { }
        public Bit64_V2(Bit64 x, Bit64 y) { X = x; Y = y; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<Bit64> GetEnumerator() { return new[] { X, Y }.Cast<Bit64>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(Bit64).Name, String.Format("({0}, {1})", X, Y)); }

        public bool Equals(Bit64_V2 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(Bit64_V2)) return false;
            return Equals((Bit64_V2)obj);
        }

        public override int GetHashCode()
        {
            unchecked
            {
                return (X.GetHashCode() * 397) ^ Y.GetHashCode();
            }
        }

        public static bool operator ==(Bit64_V2 left, Bit64_V2 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(Bit64_V2 left, Bit64_V2 right)
        {
            return !left.Equals(right);
        }
    }
}