using System;
using System.Diagnostics;

namespace Libptx.Instructions.Enumerations
{
    [DebuggerNonUserCode]
    internal class clampm
    {
        public static clampm trap { get { throw new NotImplementedException(); } }
        public static clampm clamp { get { throw new NotImplementedException(); } }
        public static clampm zero { get { throw new NotImplementedException(); } }

        public String name { get { throw new NotImplementedException(); } }
        public static bool operator ==(clampm m1, clampm m2) { throw new NotImplementedException(); }
        public static bool operator !=(clampm m1, clampm m2) { return !(m1 == m2); }
        public override bool Equals(Object obj) { throw new NotImplementedException(); }
        public override int GetHashCode() { throw new NotImplementedException(); }

        public static implicit operator String(clampm clampm) { throw new NotImplementedException(); }
        public static implicit operator clampm(String clampm) { throw new NotImplementedException(); }
    }

    // todo. implement those null-safely

    [DebuggerNonUserCode]
    internal static class clampm_extensions
    {
        public static String name(this clampm clampm) { throw new NotImplementedException(); }
    }
}