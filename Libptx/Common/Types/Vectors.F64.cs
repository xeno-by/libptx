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
    public struct double3 : IEnumerable<double>, IEquatable<double3>
    {
        public double X;
        public double Y;
        public double Z;

        public double3(double x) : this(x, default(double), default(double)) { }
        public double3(double x, double y) : this(x, y, default(double)) { }
        public double3(double x, double y, double z) { X = x; Y = y; Z = z; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<double> GetEnumerator() { return new[] { X, Y, Z }.Cast<double>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(double).Name, String.Format("({0}, {1}, {2})", X, Y, Z)); }

        public bool Equals(double3 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(double3)) return false;
            return Equals((double3)obj);
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

        public static bool operator ==(double3 left, double3 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(double3 left, double3 right)
        {
            return !left.Equals(right);
        }
    }

    [DebuggerNonUserCode]
    [StructLayout(LayoutKind.Sequential)]
    public struct double4 : IEnumerable<double>, IEquatable<double4>
    {
        public double X;
        public double Y;
        public double Z;
        public double W;

        public double4(double x) : this(x, default(double), default(double), default(double)) { }
        public double4(double x, double y) : this(x, y, default(double), default(double)) { }
        public double4(double x, double y, double z) : this(x, y, z, default(double)) { }
        public double4(double x, double y, double z, double w) { X = x; Y = y; Z = z; W = w; }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
        public IEnumerator<double> GetEnumerator() { return new[] { X, Y, Z, W }.Cast<double>().GetEnumerator(); }
        public override String ToString() { return String.Format("{0}{1}", typeof(double).Name, String.Format("({0}, {1}, {2}, {3})", X, Y, Z, W)); }

        public bool Equals(double4 other)
        {
            return Equals(other.X, X) && Equals(other.Y, Y) && Equals(other.Z, Z) && Equals(other.W, W);
        }

        public override bool Equals(Object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (obj.GetType() != typeof(double4)) return false;
            return Equals((double4)obj);
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

        public static bool operator ==(double4 left, double4 right)
        {
            return left.Equals(right);
        }

        public static bool operator !=(double4 left, double4 right)
        {
            return !left.Equals(right);
        }
    }
}